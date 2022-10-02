#include "physics_system.h"
#include <cooperative_groups.h>

#define GRAVITY 9.8f
#define PIXEL_PER_METER 100.0f
#define SMOOTH 20.0f
#define REST 0.2
#define STIFF 500000.0
#define PI 3.141592
#define SQRT_PI 1.772453
#define M_P REST*RADIUS*RADIUS*4
#define VEL_LIMIT 800.0

#define BLOCK_SIZE 96
#define SHARED_MEM_PER_THREAD 101

__global__ void updateParticles(float dt, Boundary* boundaries, int numBoundaries, Particle* particles, Particle* oldParticles, int numParticles, Pump* pumps, PumpVelocity* pumpVelocities, int numPumps, float* pressureDensityRatios);

PhysicsSystem::PhysicsSystem(GraphicsEngine& gfx, EntityManager& manager)
    :
    numBoundaries(manager.getBoundaries().size()),
    numParticles(manager.getParticles().numberOfCircles),
    numPumps(manager.getPumps().size())
{
    float refreshRate;
    if(RATE_IS_INVALID(refreshRate = gfx.getRefreshRate())){
        throw std::exception("Refreshrate could not easily be found programmatically.");
    }
    dt = 1.0f/(UPDATES_PER_RENDER*refreshRate);

    // First check whether grid sync is possible by querying the device attribute
    cudaError_t err;
    int dev = 0;
    int supportsCoopLaunch = 0;
    CUDA_THROW_FAILED(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if(supportsCoopLaunch!=1){
        throw std::exception("Cooperative Launch is not supported on the GPU, which is neccesary for the program");
    }

    // Next allocate memory on the GPU necessary for the kernel
    allocateDeviceMemory(manager);

    // Then transfer the information from the entitymanager to the GPU
    transferToDeviceMemory(manager);
}

PhysicsSystem::~PhysicsSystem(){
    destroyDeviceMemory();
}

void PhysicsSystem::update(EntityManager& manager){
    cudaError_t err;

    dim3 numBlocks((numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE);
    int sharedMemorySize = numBoundaries*sizeof(Boundary)+numPumps*(sizeof(Pump)+sizeof(PumpVelocity))+SHARED_MEM_PER_THREAD*BLOCK_SIZE;
    
    Particle* particles = (Particle*)manager.getParticles().getMappedAccess();
    void* kernelArgs[] = {&dt, &boundaries, &numBoundaries, &particles, &oldParticles, &numParticles, &pumps, &pumpVelocities, &numPumps, &pressureDensityRatios};
    CUDA_THROW_FAILED(cudaLaunchCooperativeKernel(updateParticles, numBlocks, blockSize, kernelArgs, sharedMemorySize, 0));
    manager.getParticles().unMap();
}

void PhysicsSystem::allocateDeviceMemory(EntityManager& manager){
    cudaError_t err;

    if(numBoundaries > 0){
        CUDA_THROW_FAILED(cudaMalloc((void**)&boundaries, sizeof(Boundary)*numBoundaries));
    }

    if(numParticles > 0){
        CUDA_THROW_FAILED(cudaMalloc((void**)&oldParticles, sizeof(Particle)*numParticles));
        CUDA_THROW_FAILED(cudaMalloc((void**)&pressureDensityRatios, sizeof(float)*numParticles));
    }

    if(numPumps > 0){
        CUDA_THROW_FAILED(cudaMalloc((void**)&pumps, sizeof(Pump)*numPumps));
        CUDA_THROW_FAILED(cudaMalloc((void**)&pumpVelocities, sizeof(PumpVelocity)*numPumps));
    }
}

void PhysicsSystem::transferToDeviceMemory(EntityManager& manager){
    cudaError_t err;

    if(numBoundaries > 0){
        CUDA_THROW_FAILED(cudaMemcpy(boundaries, manager.getBoundaries().data(), sizeof(Boundary)*numBoundaries, cudaMemcpyHostToDevice));
    }

    Particle* particles = (Particle*)manager.getParticles().getMappedAccess();
    if(numParticles > 0){
        CUDA_THROW_FAILED(cudaMemcpy(oldParticles, particles, sizeof(Particle)*numParticles, cudaMemcpyHostToDevice));
    }
    manager.getParticles().unMap();

    if(numPumps > 0){
        CUDA_THROW_FAILED(cudaMemcpy(pumps, manager.getPumps().data(), sizeof(Pump)*numPumps, cudaMemcpyHostToDevice));
        CUDA_THROW_FAILED(cudaMemcpy(pumpVelocities, manager.getPumpVelocities().data(), sizeof(PumpVelocity)*numPumps, cudaMemcpyHostToDevice));
    }
}

void PhysicsSystem::destroyDeviceMemory() noexcept{
    if(boundaries){
        cudaFree(boundaries);
    }

    if(oldParticles){
        cudaFree(oldParticles);
    }

    if(pumps){
        cudaFree(pumps);
    }

    if(pumpVelocities){
        cudaFree(pumpVelocities);
    }

    if(pressureDensityRatios){
        cudaFree(pressureDensityRatios);
    }
}

__global__ void updateParticles(float dt, Boundary* boundaries, int numBoundaries, Particle* particles, Particle* oldParticles, int numParticles, Pump* pumps, PumpVelocity* pumpVelocities, int numPumps, float* pressureDensityRatios){
    // Initialize shared memory pointers
    extern __shared__ Boundary boundaries_local_pointer[];
    Pump* pumps_local_pointer = (Pump*)(&boundaries_local_pointer[numBoundaries]);
    PumpVelocity* pumpvelocities_local_pointer = (PumpVelocity*)(&pumps_local_pointer[numPumps]);

    // Get the grid_group because later on device wide synchronization will be necessary
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i=threadIdx.x; i<numBoundaries; i+=blockDim.x){
        boundaries_local_pointer[i] = boundaries[i];
    }

    for(int i=threadIdx.x; i<numPumps; i+=blockDim.x){
        pumps_local_pointer[i] = pumps[i];
        pumpvelocities_local_pointer[i] = pumpVelocities[i];
    }

    // Wait until shared memory has been initialized
    __syncthreads();


    float old_x = 0.0;
    float old_y = 0.0;
    float my_x = 0.0;
    float my_y = 0.0;
    if(thread_id < numParticles){ 
        old_x = oldParticles[thread_id].x;
        old_y = oldParticles[thread_id].y;
        my_x = particles[thread_id].x;
        my_y = particles[thread_id].y;
    }

    for(int e=0; e<UPDATES_PER_RENDER; e++){
        if(thread_id < numParticles){
            float vel_x = (my_x - old_x) / dt;
            float vel_y = (my_y - old_y) / dt;

            // Update velocities based on whether the particle is in a pump or not
            for(int i = 0; i < numPumps; i++){
                if(my_x >= pumps_local_pointer[i].xLow && my_x <= pumps_local_pointer[i].xHigh && my_y >= pumps_local_pointer[i].yLow && my_y <= pumps_local_pointer[i].yHigh){
                    vel_x = pumpvelocities_local_pointer[i].velX;
                    vel_y = pumpvelocities_local_pointer[i].velY;
                    break;
                }
            }

            // Update positional change of particles caused by gravity
            vel_y -= GRAVITY*PIXEL_PER_METER*dt;

            // Update particle positions
            my_x += vel_x*dt;
            my_y += vel_y*dt;


            // Update positional change of particles caused by boundaries (make sure particles cannot pass boundaries)
            for(int i=0; i<numBoundaries; i++){
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
                my_x = crossing_x - RADIUS*line_nx/sqrt(line_nx*line_nx+line_ny*line_ny);
                my_y = crossing_y - RADIUS*line_ny/sqrt(line_nx*line_nx+line_ny*line_ny);
                vel_x = 0.0;
                vel_y = 0.0;
                break;
            }

            // Store the old particle positions
            old_x = my_x - vel_x*dt;
            old_y = my_y - vel_y*dt;

            // Also update the particle positions in global memory
            particles[thread_id].x = my_x;
            particles[thread_id].y = my_y;
        }

        // Synchronize the grid
        grid.sync();

        unsigned char* boundary_neighbour_indexes = (unsigned char*)(&pumpvelocities_local_pointer[numPumps])+threadIdx.x*SHARED_MEM_PER_THREAD;
        int number_of_boundary_neighbours = 0;
        unsigned short* particle_neighbour_indexes = NULL;
        int number_of_particle_neighbours = 0;

        float my_pressure_density_ratio = 0.0;

        float boundary_average_nx = 0.0;
        float boundary_average_ny = 0.0;

        if(thread_id < numParticles){
            // Look for boundaries near the particle
            for(unsigned char i=0; i<numBoundaries; i++){
                Boundary line = boundaries_local_pointer[i];
                float line_nx = line.y2-line.y1;
                float line_ny = line.x1-line.x2;
                float multiplier = rsqrt(line_nx*line_nx+line_ny*line_ny);
                line_nx *= multiplier;
                line_ny *= multiplier;
                float projection = (line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny;
                if(projection >= 0){
                    float crossing_x = my_x + projection*line_nx;
                    float crossing_y = my_y + projection*line_ny;
                    float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                    float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
                    if(projection <= SMOOTH && second_check3 <= (line.x2-line.x1)*(line.x2-line.x1)+(line.y2-line.y1)*(line.y2-line.y1) && second_check4 >= 0){
                        // Particle is hovering above this boundary and distance from boundary is less than SMOOTH
                        boundary_neighbour_indexes[number_of_boundary_neighbours] = i;
                        number_of_boundary_neighbours++;
                        boundary_average_nx += line_nx;
                        boundary_average_ny += line_ny;
                    }
                    else if(((line.x1-my_x)*(line.x1-my_x) + (line.y1-my_y)*(line.y1-my_y)) < SMOOTH*SMOOTH && ((line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny) > 0){
                        // Particle is close enough to one endpoint of the boundary
                        boundary_neighbour_indexes[number_of_boundary_neighbours] = i;
                        number_of_boundary_neighbours++;
                        boundary_average_nx += line_nx;
                        boundary_average_ny += line_ny;
                    }
                    else if(((line.x2-my_x)*(line.x2-my_x) + (line.y2-my_y)*(line.y2-my_y)) < SMOOTH*SMOOTH && ((line.x2-my_x)*line_nx+(line.y2-my_y)*line_ny) > 0){
                        // Particle is close enough to another endpoint of the boundary
                        boundary_neighbour_indexes[number_of_boundary_neighbours] = i;
                        number_of_boundary_neighbours++;
                        boundary_average_nx += line_nx;
                        boundary_average_ny += line_ny;
                    }
                }
            }

            // Initialize the particle neighbours pointer (conversion from a char pointer to short pointer requires alignment)
            int aligner = (long long)&boundary_neighbour_indexes[number_of_boundary_neighbours] & 1 ? 1 : 0;
            particle_neighbour_indexes = (unsigned short*)(&boundary_neighbour_indexes[number_of_boundary_neighbours]+aligner);

            float dens = 0.0;

            // Find particle neighbours
            for (unsigned short i=0; i < numParticles; i++) {
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
                            if(first_check <= 0.0){
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
                            float projection = (line.x1-p2.x)*line_nx +(line.y1-p2.y)*line_ny;
                            float virtual_x = p2.x + 2*projection*line_nx/(line_nx*line_nx+line_ny*line_ny);
                            float virtual_y = p2.y + 2*projection*line_ny/(line_nx*line_nx+line_ny*line_ny);
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
            if(dens>0.0){
                // Calculate the pressure_density_ratio
                my_pressure_density_ratio = STIFF*(dens-REST)/(dens*dens);
                pressureDensityRatios[thread_id] = my_pressure_density_ratio;
            }
        }

        // Synchronize the grid
        grid.sync();

        if(thread_id < numParticles){
            float vel_x = 0.0;
            float vel_y = 0.0;
            float boundaries_vel_x = 0.0;
            float boundaries_vel_y = 0.0;
            for(int i=0; i<number_of_particle_neighbours; i++){
                unsigned short particle_index = particle_neighbour_indexes[i];
                Particle p2 = particles[particle_index];
                float p2_pressure_density_ratio = pressureDensityRatios[particle_index];
                float press = M_P*(my_pressure_density_ratio + p2_pressure_density_ratio);

                // First calculate displacement of the particle caused by neighbours
                {
                    float dist_squared = (my_x-p2.x)*(my_x-p2.x)+(my_y-p2.y)*(my_y-p2.y);
                    float q = (float)(2.0*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                    float displace = (press * q) * dt;
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
                    float projection = (line.x1-p2.x)*line_nx +(line.y1-p2.y)*line_ny;
                    float virtual_x = p2.x + 2*projection*line_nx/(line_nx*line_nx+line_ny*line_ny);
                    float virtual_y = p2.y + 2*projection*line_ny/(line_nx*line_nx+line_ny*line_ny);
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
                    float q = (float)(2.0*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                    float displace = (press * q) * dt;
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
                float projection = (line.x1-my_x)*line_nx +(line.y1-my_y)*line_ny;
                float virtual_x = my_x + 2*projection*line_nx/(line_nx*line_nx+line_ny*line_ny);
                float virtual_y = my_y + 2*projection*line_ny/(line_nx*line_nx+line_ny*line_ny);
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
                float q = (float)(2.0*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                float displace = (M_P * 2.0 * my_pressure_density_ratio * q) * dt;
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
                float multiplier = VEL_LIMIT/sqrt(vel_x*vel_x+vel_y*vel_y);
                vel_x *= multiplier;
                vel_y *= multiplier;
            }

            my_x += vel_x*dt;
            my_y += vel_y*dt;
        }
    }

    grid.sync();

    // Store the both position and old positions in global memeory
    if(thread_id < numParticles){
        oldParticles[thread_id].x = old_x;
        oldParticles[thread_id].y = old_y;

        particles[thread_id].x = my_x;
        particles[thread_id].y = my_y;
    }
}