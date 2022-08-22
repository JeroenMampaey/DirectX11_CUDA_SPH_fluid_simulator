#include "physics.h"
#include <vector>
#include <string>
#include <cooperative_groups.h>

//TODO:
//  -Try using register hints, 8.6 compute capability should offer me about 80 32-bit registers
//  -Compare Boundary &line = ... performance to Boundary line = ...
//  -When adding the ghost particles for each neighbour, eliminate the extra scopes by using sensible names instead of first_check etc.
//  -When checking whether particle crosses this boundary: first_check<0 or first_check<=0??
//  -if(dens==0.0) dens=0.01; is still very arbitrary
//  -is a for-loop necessary for checking if boundaries_vel is along the average boundary normal?
//  -include your own thread_id when making particle movement changes

// Private declared functions
__global__ void updateParticles(Boundary* boundaries, int numboundaries, Particle* particles, Particle* old_particles, int numpoints, Pump* pumps, PumpVelocity* pumpvelocities, int numpumps, float* pressure_density_ratios);
cudaError_t allocateDeviceMemory(Boundary* &device_boundaries, int numboundaries, Particle* &device_particles, Particle* &old_particles, int numpoints, Pump* &device_pumps, PumpVelocity* &device_pumpvelocities, int numpumps, float* &pressure_density_ratios);
void destroyDeviceMemory(Boundary* device_boundaries, int numboundaries, Particle* device_particles, Particle* old_particles, int numpoints, Pump* device_pumps, PumpVelocity* device_pumpvelocities, int numpumps, float* pressure_density_ratios);
cudaError_t transferToDeviceMemory(Boundary* boundaries, Boundary* device_boundaries, int numboundaries, Particle* particles, Particle* device_particles, Particle* old_particles, int numpoints, Pump* pumps, Pump* device_pumps, PumpVelocity* pumpvelocities, PumpVelocity* device_pumpvelocities, int numpumps);
cudaError_t setupDeviceMemory(Boundary* boundaries, Boundary* &device_boundaries, int numboundaries, Particle* particles, Particle* &device_particles, Particle* &old_particles, int numpoints, Pump* pumps, Pump* &device_pumps, PumpVelocity* pumpvelocities, PumpVelocity* &device_pumpvelocities, int numpumps, float* &pressure_density_ratios);

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<bool> &doneDrawing, Boundary* boundaries, int numboundaries, Particle* particles, int numpoints, Pump* pumps, PumpVelocity* pumpvelocities, int numpumps, HWND m_hwnd){
    Boundary* device_boundaries = NULL;
    Particle* device_particles = NULL;
    Particle* old_particles = NULL;
    Pump* device_pumps = NULL;
    PumpVelocity* device_pumpvelocities = NULL;
    float* pressure_density_ratios = NULL;

    cudaError_t success = setupDeviceMemory(boundaries, device_boundaries, numboundaries, particles, device_particles, old_particles, numpoints, pumps, device_pumps, pumpvelocities, device_pumpvelocities, numpumps, pressure_density_ratios);

    while(!exit.load()){
        bool expected = true;
        if(updateRequired.compare_exchange_weak(expected, false)){
            // If an update is necessary, update the particles UPDATES_PER_RENDER times and then redraw the particles
            if(success==cudaSuccess){
                dim3 numBlocks((numpoints + BLOCK_SIZE - 1) / BLOCK_SIZE);
                dim3 blockSize(BLOCK_SIZE);
                int sharedMemorySize = numboundaries*sizeof(Boundary)+numpumps*sizeof(Pump)+numpumps*sizeof(PumpVelocity)+SHARED_MEM_PER_THREAD*BLOCK_SIZE;
                //updateParticles<<<numBlocks, BLOCK_SIZE, sharedMemorySize>>>(device_boundaries, numboundaries, device_particles, old_particles, numpoints, device_pumps, device_pumpvelocities, numpumps, pressure_density_ratios);
                void* kernelArgs[] = {&device_boundaries, &numboundaries, &device_particles, &old_particles, &numpoints, &device_pumps, &device_pumpvelocities, &numpumps, &pressure_density_ratios};
                cudaLaunchCooperativeKernel(updateParticles, numBlocks, blockSize, kernelArgs, sharedMemorySize, 0);
                cudaError_t err = cudaGetLastError();
                if(DEBUG && err!=cudaSuccess){
                    Debugger::updateTopic(DEBUG_THREAD2_ERROR, cudaGetErrorString(err));
                }
                while(!doneDrawing.load()){}
                cudaMemcpy(particles, device_particles, sizeof(Particle) * numpoints, cudaMemcpyDeviceToHost);
            }
            doneDrawing.store(false);

            // Redraw the particles
            InvalidateRect(m_hwnd, NULL, FALSE);
        }
    }

    destroyDeviceMemory(device_boundaries, numboundaries, device_particles, old_particles, numpoints, device_pumps, device_pumpvelocities, numpumps, pressure_density_ratios);
}

__global__ void updateParticles(Boundary* boundaries, int numboundaries, Particle* particles, Particle* old_particles, int numpoints, Pump* pumps, PumpVelocity* pumpvelocities, int numpumps, float* pressure_density_ratios){
    extern __shared__ Boundary s[];

    // Get the grid_group because later on device wide synchronization will be necessary
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    // Put all the boundaries and all the pumps in shared memory
    Boundary* boundaries_local_pointer = s;
    Pump* pumps_local_pointer = (Pump*)(&s[numboundaries]);
    PumpVelocity* pumpvelocities_local_pointer = (PumpVelocity*)(&pumps_local_pointer[numpumps]);

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i=threadIdx.x; i<numboundaries; i+=blockDim.x){
        boundaries_local_pointer[i] = boundaries[i];
    }

    for(int i=threadIdx.x; i<numpumps; i+=blockDim.x){
        pumps_local_pointer[i] = pumps[i];
        pumpvelocities_local_pointer[i] = pumpvelocities[i];
    }

    // Wait until shared memory has been initialized
    __syncthreads();


    float old_x = 0.0;
    float old_y = 0.0;
    float my_x = 0.0;
    float my_y = 0.0;
    if(thread_id < numpoints){ 
        old_x = old_particles[thread_id].x;
        old_y = old_particles[thread_id].y;
        my_x = particles[thread_id].x;
        my_y = particles[thread_id].y;
    }

    for(int e=0; e<UPDATES_PER_RENDER; e++){
        if(thread_id < numpoints){
            float vel_x = (my_x - old_x) / (INTERVAL_MILI/1000.0);
            float vel_y = (my_y - old_y) / (INTERVAL_MILI/1000.0);

            // Update velocities based on whether the particle is in a pump or not
            for(int i = 0; i < numpumps; i++){
                if(my_x >= pumps_local_pointer[i].x_low && my_x <= pumps_local_pointer[i].x_high && my_y >= pumps_local_pointer[i].y_low && my_y <= pumps_local_pointer[i].y_high){
                    vel_x = pumpvelocities_local_pointer[i].velx;
                    vel_y = pumpvelocities_local_pointer[i].vely;
                    break;
                }
            }

            // Update positional change of particles caused by gravity
            vel_y += GRAVITY*PIXEL_PER_METER*(INTERVAL_MILI/1000.0);

            // Update positional change of particles caused boundaries (make sure particles cannot pass boundaries)
            for(int i=0; i<numboundaries; i++){
                Boundary line = boundaries_local_pointer[i];
                float line_nx = line.y2-line.y1;
                float line_ny = line.x1-line.x2;
                float first_check = ((my_x-line.x1)*line_nx+(my_y-line.y1)*line_ny)*((old_x-line.x1)*line_nx+(old_y-line.y1)*line_ny);
                if(first_check > 0) continue;
                float second_check1 = (line.x1-old_x)*line_nx+(line.y1-old_y)*line_ny;
                if(second_check1 < 0) continue;
                float second_check2 = (my_x-old_x)*line_nx+(my_y-old_y)*line_ny;
                float crossing_x = old_x;
                float crossing_y = old_y;
                if(second_check2 > 0.0){
                    crossing_x += (my_x-old_x)*second_check1/second_check2;
                    crossing_y += (my_y-old_y)*second_check1/second_check2;
                }
                float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
                if(second_check3>(line.x2-line.x1)*(line.x2-line.x1)+(line.y2-line.y1)*(line.y2-line.y1) || second_check4<0.0) continue;
                my_x = crossing_x - (RADIUS/2)*line_nx/sqrtf(line_nx*line_nx+line_ny*line_ny);
                my_y = crossing_y - (RADIUS/2)*line_ny/sqrtf(line_nx*line_nx+line_ny*line_ny);
                vel_x = 0;
                vel_y = 0;
                break;
            }

            // Update particle positions
            old_x = my_x;
            old_y = my_y;
            my_x += vel_x*(INTERVAL_MILI/1000.0);
            my_y += vel_y*(INTERVAL_MILI/1000.0);

            // Also update the particle positions in global memory
            particles[thread_id] = {my_x, my_y};
        }

        // Synchronize the grid
        grid.sync();

        unsigned char* boundary_neighbour_indexes = (unsigned char*)(&pumpvelocities_local_pointer[numpumps])+threadIdx.x*SHARED_MEM_PER_THREAD;
        int number_of_boundary_neighbours = 0;
        unsigned short* particle_neighbour_indexes = NULL;
        int number_of_particle_neighbours = 0;

        float my_pressure_density_ratio = 0.0;

        float boundary_average_nx = 0.0;
        float boundary_average_ny = 0.0;

        if(thread_id < numpoints){
            // Look for boundaries near the particle
            for(unsigned char i=0; i<numboundaries; i++){
                Boundary line = boundaries_local_pointer[i];
                float line_nx = line.y2-line.y1;
                float line_ny = line.x1-line.x2;
                line_nx *= 1/sqrtf(line_nx*line_nx+line_ny*line_ny);
                line_ny *= 1/sqrtf(line_nx*line_nx+line_ny*line_ny);
                float projection = (line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny;
                if(projection >= 0){
                    float crossing_x = my_x + projection*line_nx;
                    float crossing_y = my_y + projection*line_ny;
                    float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                    float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
                    bool particle_is_near_to_line = projection <= SMOOTH && second_check3 <= (line.x2-line.x1)*(line.x2-line.x1)+(line.y2-line.y1)*(line.y2-line.y1) && second_check4 >= 0;
                    bool particle_is_near_to_endpoint1 = ((line.x1-my_x)*(line.x1-my_x) + (line.y1-my_y)*(line.y1-my_y)) < SMOOTH*SMOOTH && ((line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny) > 0;
                    bool particle_is_near_to_endpoint2 = ((line.x2-my_x)*(line.x2-my_x) + (line.y2-my_y)*(line.y2-my_y)) < SMOOTH*SMOOTH && ((line.x2-my_x)*line_nx+(line.y2-my_y)*line_ny) > 0;
                    if(particle_is_near_to_line || particle_is_near_to_endpoint1 || particle_is_near_to_endpoint2){
                        // Particle is near enough to the boundary
                        boundary_neighbour_indexes[number_of_boundary_neighbours] = i;
                        number_of_boundary_neighbours++;
                        boundary_average_nx += line_nx;
                        boundary_average_ny += line_ny;
                    }
                }
            }

            // Initialize the particle neighbours pointer
            int aligner = (long long)&boundary_neighbour_indexes[number_of_boundary_neighbours] & 1 ? 1 : 0;
            particle_neighbour_indexes = (unsigned short*)(&boundary_neighbour_indexes[number_of_boundary_neighbours]+aligner);

            float dens = 0.0;

            // Find particle neighbours
            for (unsigned short i=0; i < numpoints; i++) {
                Particle p2 = particles[i];
                float dist_squared = (my_x-p2.x)*(my_x-p2.x)+(my_y-p2.y)*(my_y-p2.y);
                if (dist_squared < SMOOTH*SMOOTH) {
                    // If the other particle is close enough, iterate over the closeby boundaries to achieve two things:
                    //  - Convert this particle to a ghost particle over the boundary
                    //  - Check whether the connection between this particle and the particle of the thread crosses a boundary
                    //    in which case the particle is not actually a true neighbour
                    float accumulated_ghost_particle_density = 0.0;
                    int j=0;
                    for(; j<number_of_boundary_neighbours; j++){
                        Boundary line = boundaries_local_pointer[boundary_neighbour_indexes[j]];
                        float line_nx = line.y2-line.y1;
                        float line_ny = line.x1-line.x2;
                        
                        // Check whether particle crosses this boundary
                        {
                            float first_check = ((my_x-line.x1)*line_nx+(my_y-line.y1)*line_ny)*((p2.x-line.x1)*line_nx+(p2.y-line.y1)*line_ny);
                            if(first_check < 0){
                                float second_check1 = (line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny;
                                float second_check2 = (p2.x-my_x)*line_nx+(p2.y-my_y)*line_ny;
                                float crossing_x = my_x;
                                float crossing_y = my_y;
                                if(second_check2 > 0.0){
                                    crossing_x += (p2.x-my_x)*second_check1/second_check2;
                                    crossing_y += (p2.y-my_y)*second_check1/second_check2;
                                }
                                float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                                float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
                                if(second_check3<=(line.x2-line.x1)*(line.x2-line.x1)+(line.y2-line.y1)*(line.y2-line.y1) && second_check4>=0.0) break;
                            }
                        }

                        // Create a ghost particle over the boundary corresponding to this neighbour
                        {
                            line_nx *= 1/sqrtf(line_nx*line_nx+line_ny*line_ny);
                            line_ny *= 1/sqrtf(line_nx*line_nx+line_ny*line_ny);
                            float projection = (line.x1-p2.x)*line_nx +(line.y1-p2.y)*line_ny;
                            float virtual_x = p2.x + 2*projection*line_nx;
                            float virtual_y = p2.y + 2*projection*line_ny;
                            float first_check = ((my_x-line.x1)*line_nx+(my_y-line.y1)*line_ny)*((virtual_x-line.x1)*line_nx+(virtual_y-line.y1)*line_ny);
                            if(first_check > 0) continue;
                            float second_check1 = (line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny;
                            float second_check2 = (virtual_x-my_x)*line_nx+(virtual_y-my_y)*line_ny;
                            float crossing_x = my_x;
                            float crossing_y = my_y;
                            if(second_check2 > 0.0){
                                crossing_x += (virtual_x-my_x)*second_check1/second_check2;
                                crossing_y += (virtual_y-my_y)*second_check1/second_check2;
                            }
                            float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                            float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
                            if(second_check3>(line.x2-line.x1)*(line.x2-line.x1)+(line.y2-line.y1)*(line.y2-line.y1) || second_check4<0.0) continue;
                            float dist_squared = (virtual_x-my_x)*(virtual_x-my_x)+(virtual_y-my_y)*(virtual_y-my_y);
                            if(dist_squared > SMOOTH*SMOOTH) continue;
                            float q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist_squared / (SMOOTH*SMOOTH/4)));
                            accumulated_ghost_particle_density += M_P*q2;
                        }
                    }

                    if(j<number_of_boundary_neighbours) continue;
                    
                    // Change the density caused by ghost particles
                    dens += accumulated_ghost_particle_density;

                    if(i==thread_id) continue;

                    // Change the density because of the neighbour particle and also add the particle to the neighbours list
                    float q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist_squared / (SMOOTH*SMOOTH/4)));
                    dens += M_P*q2;
                    particle_neighbour_indexes[number_of_particle_neighbours] = i;
                    number_of_particle_neighbours++;
                }
            }
            
            // Make sure no division by zero exceptions occur
            //if(dens<=0.0) dens=0.00001; 

            // Calculate the pressure_density_ratio
            my_pressure_density_ratio = STIFF*(dens-REST)/(dens*dens);
            pressure_density_ratios[thread_id] = my_pressure_density_ratio;
        }

        // Synchronize the grid
        grid.sync();

        if(thread_id < numpoints){
            float vel_x = 0.0;
            float vel_y = 0.0;
            float boundaries_vel_x = 0.0;
            float boundaries_vel_y = 0.0;
            for(int i=0; i<number_of_particle_neighbours; i++){
                unsigned short particle_index = particle_neighbour_indexes[i];
                Particle p2 = particles[particle_index];
                float p2_pressure_density_ratio = pressure_density_ratios[particle_index];
                float press = M_P*(my_pressure_density_ratio + p2_pressure_density_ratio);

                // First calculate displacement of the particle caused by neighbours
                {
                    float dist_squared = (my_x-p2.x)*(my_x-p2.x)+(my_y-p2.y)*(my_y-p2.y);
                    float q = (float)(2*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                    float displace = (press * q) * (INTERVAL_MILI/1000.0);
                    float abx = (my_x - p2.x);
                    float aby = (my_y - p2.y);
                    vel_x += displace * abx;
                    vel_y += displace * aby;
                }

                // Next calculate displacement of the particle caused by boundaries
                for(int j=0; j<number_of_boundary_neighbours; j++){
                    Boundary line = boundaries_local_pointer[boundary_neighbour_indexes[j]];
                    float line_nx = (line.y2-line.y1);
                    float line_ny = (line.x1-line.x2);
                    line_nx *= 1/sqrtf(line_nx*line_nx+line_ny*line_ny);
                    line_ny *= 1/sqrtf(line_nx*line_nx+line_ny*line_ny);
                    float projection = (line.x1-p2.x)*line_nx +(line.y1-p2.y)*line_ny;
                    float virtual_x = p2.x + 2*projection*line_nx;
                    float virtual_y = p2.y + 2*projection*line_ny;
                    float first_check = ((my_x-line.x1)*line_nx+(my_y-line.y1)*line_ny)*((virtual_x-line.x1)*line_nx+(virtual_y-line.y1)*line_ny);
                    if(first_check > 0) continue;
                    float second_check1 = (line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny;
                    float second_check2 = (virtual_x-my_x)*line_nx+(virtual_y-my_y)*line_ny;
                    float crossing_x = my_x;
                    float crossing_y = my_y;
                    if(second_check2 > 0.0){
                        crossing_x += (virtual_x-my_x)*second_check1/second_check2;
                        crossing_y += (virtual_y-my_y)*second_check1/second_check2;
                    }
                    float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                    float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
                    if(second_check3>(line.x2-line.x1)*(line.x2-line.x1)+(line.y2-line.y1)*(line.y2-line.y1) || second_check4<0.0) continue;
                    float dist_squared = (virtual_x-my_x)*(virtual_x-my_x)+(virtual_y-my_y)*(virtual_y-my_y);
                    if(dist_squared > SMOOTH*SMOOTH) continue;
                    float q = (float)(2*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                    float displace = (press * q) * (INTERVAL_MILI/1000.0);
                    float abx = (my_x - virtual_x);
                    float aby = (my_y - virtual_y);
                    boundaries_vel_x += displace * abx;
                    boundaries_vel_y += displace * aby;
                }
            }

            // Also include ghost particles made by the particle itself
            for(int j=0; j<number_of_boundary_neighbours; j++){
                Boundary line = boundaries_local_pointer[boundary_neighbour_indexes[j]];
                float line_nx = (line.y2-line.y1);
                float line_ny = (line.x1-line.x2);
                line_nx *= 1/sqrtf(line_nx*line_nx+line_ny*line_ny);
                line_ny *= 1/sqrtf(line_nx*line_nx+line_ny*line_ny);
                float projection = (line.x1-my_x)*line_nx +(line.y1-my_y)*line_ny;
                float virtual_x = my_x + 2*projection*line_nx;
                float virtual_y = my_y + 2*projection*line_ny;
                float first_check = ((my_x-line.x1)*line_nx+(my_y-line.y1)*line_ny)*((virtual_x-line.x1)*line_nx+(virtual_y-line.y1)*line_ny);
                if(first_check > 0) continue;
                float second_check1 = (line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny;
                float second_check2 = (virtual_x-my_x)*line_nx+(virtual_y-my_y)*line_ny;
                float crossing_x = my_x;
                float crossing_y = my_y;
                if(second_check2 > 0.0){
                    crossing_x += (virtual_x-my_x)*second_check1/second_check2;
                    crossing_y += (virtual_y-my_y)*second_check1/second_check2;
                }
                float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
                if(second_check3>(line.x2-line.x1)*(line.x2-line.x1)+(line.y2-line.y1)*(line.y2-line.y1) || second_check4<0.0) continue;
                float dist_squared = (virtual_x-my_x)*(virtual_x-my_x)+(virtual_y-my_y)*(virtual_y-my_y);
                if(dist_squared > SMOOTH*SMOOTH) continue;
                float q = (float)(2*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                float displace = (M_P * 2 * my_pressure_density_ratio * q) * (INTERVAL_MILI/1000.0);
                float abx = (my_x - virtual_x);
                float aby = (my_y - virtual_y);
                boundaries_vel_x += displace * abx;
                boundaries_vel_y += displace * aby;
            }

            // Only allow a boundary to cause a velocity change if the particle is repulsed by it 
            // since boundaries should not be able to attract particles
            if(boundary_average_nx*boundaries_vel_x+boundary_average_ny*boundaries_vel_y <=0.0){
                vel_x += boundaries_vel_x;
                vel_y += boundaries_vel_y;
            }

            // Put a velocity limit on the particles too allow the system to work still somewhat normally 
            // if some unforeseen behaviour would occur
            if(vel_x*vel_x+vel_y*vel_y > VEL_LIMIT*VEL_LIMIT){
                vel_x *= VEL_LIMIT/sqrtf(vel_x*vel_x+vel_y*vel_y);
                vel_y *= VEL_LIMIT/sqrtf(vel_x*vel_x+vel_y*vel_y);
            }

            my_x += vel_x*(INTERVAL_MILI/1000.0);
            my_y += vel_y*(INTERVAL_MILI/1000.0);
        }
    }

    grid.sync();

    // Store the both position and old positions in global memeory
    if(thread_id < numpoints){
        old_particles[thread_id] = {old_x, old_y};
        particles[thread_id] = {my_x, my_y};
    }
}

void destroyDeviceMemory(Boundary* device_boundaries, int numboundaries, Particle* device_particles, Particle* old_particles, int numpoints, Pump* device_pumps, PumpVelocity* device_pumpvelocities, int numpumps, float* pressure_density_ratios){
    if(device_boundaries){
        cudaFree(device_boundaries);
    }

    if(device_particles){
        cudaFree(device_particles);
    }

    if(old_particles){
        cudaFree(old_particles);
    }

    if(device_pumps){
        cudaFree(device_pumps);
    }

    if(device_pumpvelocities){
        cudaFree(device_pumpvelocities);
    }

    if(pressure_density_ratios){
        cudaFree(pressure_density_ratios);
    }
}

cudaError_t allocateDeviceMemory(Boundary* &device_boundaries, int numboundaries, Particle* &device_particles, Particle* &old_particles, int numpoints, Pump* &device_pumps, PumpVelocity* &device_pumpvelocities, int numpumps, float* &pressure_density_ratios){
    cudaError_t status;

    if(numboundaries > 0){
        status = cudaMalloc((void**)&device_boundaries, sizeof(Boundary) * numboundaries);
        if(status != cudaSuccess){
            return status;
        }
    }

    if(numpoints > 0){
        status = cudaMalloc((void**)&device_particles, sizeof(Particle) * numpoints);
        if(status != cudaSuccess){
            return status;
        }
        status = cudaMalloc((void**)&old_particles, sizeof(Particle) * numpoints);
        if(status != cudaSuccess){
            return status;
        }
        status = cudaMalloc((void**)&pressure_density_ratios, sizeof(float) * numpoints);
        if(status != cudaSuccess){
            return status;
        }
    }

    if(numpumps > 0){
        status = cudaMalloc((void**)&device_pumps, sizeof(Pump) * numpumps);
        if(status != cudaSuccess){
            return status;
        }
        status = cudaMalloc((void**)&device_pumpvelocities, sizeof(PumpVelocity) * numpumps);
        if(status != cudaSuccess){
            return status;
        }
    }

    return cudaSuccess;
}

cudaError_t transferToDeviceMemory(Boundary* boundaries, Boundary* device_boundaries, int numboundaries, Particle* particles, Particle* device_particles, Particle* old_particles, int numpoints, Pump* pumps, Pump* device_pumps, PumpVelocity* pumpvelocities, PumpVelocity* device_pumpvelocities, int numpumps){
    cudaError_t status;

    if(numboundaries > 0){
        status = cudaMemcpy(device_boundaries, boundaries, sizeof(Boundary) * numboundaries, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return status;
        }
    }

    if(numpoints > 0){
        status = cudaMemcpy(device_particles, particles, sizeof(Particle) * numpoints, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return status;
        }

        status = cudaMemcpy(old_particles, particles, sizeof(Particle) * numpoints, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return status;
        }
    }

    if(numpumps > 0){
        status = cudaMemcpy(device_pumps, pumps, sizeof(Pump) * numpumps, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return status;
        }

        status = cudaMemcpy(device_pumpvelocities, pumpvelocities, sizeof(PumpVelocity) * numpumps, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return status;
        }
    }

    return cudaSuccess;
}

cudaError_t setupDeviceMemory(Boundary* boundaries, Boundary* &device_boundaries, int numboundaries, Particle* particles, Particle* &device_particles, Particle* &old_particles, int numpoints, Pump* pumps, Pump* &device_pumps, PumpVelocity* pumpvelocities, PumpVelocity* &device_pumpvelocities, int numpumps, float* &pressure_density_ratios){
    cudaError_t success = allocateDeviceMemory(device_boundaries, numboundaries, device_particles, old_particles, numpoints, device_pumps, device_pumpvelocities, numpumps, pressure_density_ratios);
    if(success!=cudaSuccess){
        if(DEBUG){
            Debugger::updateTopic(DEBUG_THREAD2_ERROR, ("Physics thread CUDA memory allocations failed, status: "+std::to_string(success)).c_str());
        }
        return success;
    }

    success = transferToDeviceMemory(boundaries, device_boundaries, numboundaries, particles, device_particles, old_particles, numpoints, pumps, device_pumps, pumpvelocities, device_pumpvelocities, numpumps);
    if(success!=cudaSuccess){
        if(DEBUG){
            Debugger::updateTopic(DEBUG_THREAD2_ERROR, ("Physics thread CUDA memory transfer failed, status: "+std::to_string(success)).c_str());
        }
        return success;
    }

    // Next, first check whether grid sync is possible by querying the device attribute
    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    if(supportsCoopLaunch!=1){
        if(DEBUG){
            Debugger::updateTopic(DEBUG_THREAD2_ERROR, "CUDA grid synchronization is not supported by the device");
        }
        return cudaErrorNotSupported;
    }

    return cudaSuccess;
}