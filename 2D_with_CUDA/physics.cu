#include "physics.h"
#include <vector>
#include <string>

// Private declared functions
__global__ void updateParticles(Boundary* boundaries, int numboundaries, Particle* particles, Particle* old_particles, int numpoints, Pump* pumps, PumpVelocity* pumpvelocities, int numpumps, float* pressure_density_ratios);
bool allocateDeviceMemory(Boundary* device_boundaries, int numboundaries, Particle* device_particles, Particle* old_particles, int numpoints, Pump* device_pumps, PumpVelocity* device_pumpvelocities, int numpumps, float* pressure_density_ratios);
void destroyDeviceMemory(Boundary* device_boundaries, int numboundaries, Particle* device_particles, Particle* old_particles, int numpoints, Pump* device_pumps, PumpVelocity* device_pumpvelocities, int numpumps, float* pressure_density_ratios);
bool transferToDeviceMemory(Boundary* boundaries, Boundary* device_boundaries, int numboundaries, Particle* particles, Particle* device_particles, Particle* old_particles, int numpoints, Pump* pumps, Pump* device_pumps, PumpVelocity* pumpvelocities, PumpVelocity* device_pumpvelocities, int numpumps);

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<bool> &doneDrawing, Boundary* boundaries, int numboundaries, Particle* particles, int numpoints, Pump* pumps, PumpVelocity* pumpvelocities, int numpumps, HWND m_hwnd){
    int blockSize = 96;
    int numBlocks = (numpoints + blockSize - 1) / blockSize;
    
    Boundary* device_boundaries;
    Particle* device_particles;
    Particle* old_particles;
    Pump* device_pumps;
    PumpVelocity* device_pumpvelocities;
    float* pressure_density_ratios;

    bool success = allocateDeviceMemory(device_boundaries, numboundaries, device_particles, old_particles, numpoints, device_pumps, device_pumpvelocities, numpumps, pressure_density_ratios);
    if(success){
        success = transferToDeviceMemory(boundaries, device_boundaries, numboundaries, particles, device_particles, old_particles, numpoints, pumps, device_pumps, pumpvelocities, device_pumpvelocities, numpumps);
    }

    while(!exit.load()){
        bool expected = true;
        if(updateRequired.compare_exchange_weak(expected, false)){
            // If an update is necessary, update the particles UPDATES_PER_RENDER times and then redraw the particles
            if(success){
                int sharedMemorySize = numboundaries*sizeof(Boundary)+numpumps*sizeof(Pump)+numpumps*sizeof(PumpVelocity)+SHARED_MEM_PER_THREAD*blockSize;
                updateParticles<<<numBlocks, blockSize, sharedMemorySize>>>(device_boundaries, numboundaries, device_particles, old_particles, numpoints, device_pumps, device_pumpvelocities, numpumps, pressure_density_ratios);
                while(!doneDrawing.load()){}
                cudaMemcpy(boundaries, device_boundaries, sizeof(Boundary) * numboundaries, cudaMemcpyDeviceToHost);
                cudaMemcpy(particles, device_particles, numpoints * sizeof(Particle), cudaMemcpyDeviceToHost);
                cudaMemcpy(pumps, device_pumps, numpumps * sizeof(Pump), cudaMemcpyDeviceToHost);
                cudaMemcpy(pumpvelocities, device_pumpvelocities, numpumps * sizeof(PumpVelocity), cudaMemcpyDeviceToHost);
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
    
    Boundary* boundaries_local_pointer = s;
    Pump* pumps_local_pointer = (Pump*)(&s[numboundaries]);
    PumpVelocity* pumpvelocities_local_pointer = (PumpVelocity*)(&pumps_local_pointer[numpumps]);
    char* thread_local_memory = (char*)(&pumpvelocities_local_pointer[numpumps]);

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i=threadIdx.x; i<numboundaries; i+=blockDim.x){
        boundaries_local_pointer[i] = boundaries[i];
    }

    for(int i=threadIdx.x; i<numpumps; i+=blockDim.x){
        pumps_local_pointer[i] = pumps[i];
        pumpvelocities_local_pointer[i] = pumpvelocities[i];
    }

    __syncthreads();


    float old_x = 0.0;
    float old_y = 0.0;
    if(thread_id < numpoints){
        old_x = old_particles[thread_id].x;
        old_y = old_particles[thread_id].y;
    }
    for(int i=0; i<UPDATES_PER_RENDER; i++){
        if(thread_id < numpoints){
            float my_x = particles[thread_id].x;
            float my_y = particles[thread_id].y;

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
                Boundary &line = boundaries_local_pointer[i];
                //TODO
                /*
                float first_check = ((p.x-line.x1)*line.nx+(p.y-line.y1)*line.ny)*((p.oldx-line.x1)*line.nx+(p.oldy-line.y1)*line.ny);
                if(first_check > 0) continue;
                float second_check1 = (line.x1-p.oldx)*line.nx+(line.y1-p.oldy)*line.ny;
                if(second_check1 < 0) continue;
                float second_check2 = (p.x-p.oldx)*line.nx+(p.y-p.oldy)*line.ny;
                float crossing_x = p.x;
                float crossing_y = p.y;
                if(second_check2 > 0.0){
                    crossing_x = p.oldx+(p.x-p.oldx)*second_check1/second_check2;
                    crossing_y = p.oldy+(p.y-p.oldy)*second_check1/second_check2;
                }
                float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
                if(second_check3>line.length_squared || second_check4<0.0) continue;
                p.x = crossing_x - 3.0*line.nx;
                p.y = crossing_y - 3.0*line.ny;
                p.velx = 0;
                p.vely = 0;
                return;*/
            }
        }
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

bool allocateDeviceMemory(Boundary* device_boundaries, int numboundaries, Particle* device_particles, Particle* old_particles, int numpoints, Pump* device_pumps, PumpVelocity* device_pumpvelocities, int numpumps, float* pressure_density_ratios){
    cudaError_t status;

    if(numboundaries > 0){
        status = cudaMalloc((void**)&device_boundaries, sizeof(Boundary) * numboundaries);
        if(status != cudaSuccess){
            return false;
        }
    }

    if(numpoints > 0){
        status = cudaMalloc((void**)&device_particles, sizeof(Particle) * numpoints);
        if(status != cudaSuccess){
            return false;
        }
        status = cudaMalloc((void**)&old_particles, sizeof(Particle) * numpoints);
        if(status != cudaSuccess){
            return false;
        }
        status = cudaMalloc((void**)&pressure_density_ratios, sizeof(float) * numpoints);
        if(status != cudaSuccess){
            return false;
        }
    }

    if(numpumps > 0){
        status = cudaMalloc((void**)&device_pumps, sizeof(Pump) * numpumps);
        if(status != cudaSuccess){
            return false;
        }
        status = cudaMalloc((void**)&device_pumpvelocities, sizeof(PumpVelocity) * numpumps);
        if(status != cudaSuccess){
            return false;
        }
    }

    return true;
}

bool transferToDeviceMemory(Boundary* boundaries, Boundary* device_boundaries, int numboundaries, Particle* particles, Particle* device_particles, Particle* old_particles, int numpoints, Pump* pumps, Pump* device_pumps, PumpVelocity* pumpvelocities, PumpVelocity* device_pumpvelocities, int numpumps){
    cudaError_t status;

    if(numboundaries > 0){
        status = cudaMemcpy(device_boundaries, boundaries, sizeof(Boundary) * numboundaries, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return false;
        }
    }

    if(numpoints > 0){
        status = cudaMemcpy(device_particles, particles, sizeof(Particle) * numpoints, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return false;
        }

        status = cudaMemcpy(old_particles, particles, sizeof(Particle) * numpoints, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return false;
        }
    }

    if(numpumps > 0){
        status = cudaMemcpy(device_pumps, pumps, sizeof(Pump) * numpumps, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return false;
        }

        status = cudaMemcpy(device_pumpvelocities, pumpvelocities, sizeof(PumpVelocity) * numpumps, cudaMemcpyHostToDevice);
        if(status != cudaSuccess){
            return false;
        }
    }

    return true;
}