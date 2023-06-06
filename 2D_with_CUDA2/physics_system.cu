#include "physics_system.h"
#include <cooperative_groups.h>
//#include <thrust/sort.h>
#include <float.h>
#include <limits.h>
/* TODO: DURATION TESTING
#include <chrono>
*/
#include <chrono>

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cub/cub.cuh>

#define GRAVITY 9.8f
#define PIXEL_PER_METER 100.0f
#define SMOOTH (3.0f*RADIUS)
#define REST 1
#define STIFF 50000.0
#define M_P (REST*RADIUS*RADIUS*PI)
#define VEL_LIMIT 8.0f

#define BLOCK_SIZE 64
#define WARP_SIZE 32

#define laneId (threadIdx.x % WARP_SIZE)
#define warpId (threadIdx.x / WARP_SIZE)

#define MAX_BOUNDARIES 100
#define MAX_PUMPS 15

// TODO
#define MAX_PARTICLE_NEIGHBOURS 40
#define MAX_BOUNDARY_NEIGHBOURS 7

#define MAX_NEARBY_PARTICLES 60
#define MAX_NEARBY_BOUNDARIES 7

/* TODO: interesting optimization
#define BLOCK_DELTA (5*SMOOTH)
*/
#define BLOCK_DELTA SMOOTH

/* TODO: kernel testing
__device__ int someMax[1] = {-1};
*/

__constant__ Boundary boundaries[MAX_BOUNDARIES];
__constant__ Pump pumps[MAX_PUMPS];
__constant__ PumpVelocity pumpVelocities[MAX_PUMPS];

struct SharedMem1{
    float particleXValues[BLOCK_SIZE];
    float particleYValues[BLOCK_SIZE];
    float particlePressureDensityRatios[BLOCK_SIZE];
    unsigned char nearbyBoundaryIndices[BLOCK_SIZE*MAX_NEARBY_BOUNDARIES];
};

__global__ void initializeParticles(Particle* graphicsParticles, float* particleXValues, float* particleYValues, float* oldParticleXValues, float* oldParticleYValues, int numParticles, unsigned short* staticIndexes);
__global__ void copyParticlesToGraphics(Particle* graphicsParticles, float* particleXValues, float* particleYValues, int numParticles);
__global__ void updateRegularPhysics(float dt, float* particleXValues, float* particleYValues, float* oldParticleXValues, float* oldParticleYValues, int numParticles, int numBoundaries, int numPumps);
__global__ void updateDensityField(float dt, 
    float* particleXValues, 
    float* particleYValues, 
    float* particlePressureDensityRatios,
    int numParticles, 
    int numBoundaries,
    unsigned char* numNearbyBoundaries,
    unsigned char* nearbyBoundaryIndices,
    unsigned char* numNearbyParticles,
    unsigned short* nearbyParticleIndices,
    int* minBlockIterator,
    int* maxBlockIterator);
__global__ void updateParticlesByDensityField(float dt, 
    float* particleXValues, 
    float* particleYValues, 
    float* particlePressureDensityRatios,
    int numParticles,
    unsigned char* numNearbyBoundaries,
    unsigned char* nearbyBoundaryIndices,
    unsigned char* numNearbyParticles,
    unsigned short* nearbyParticleIndices,
    int* minBlockIterator,
    int* maxBlockIterator);
__global__ void permuteArrays(unsigned short* permutedIndices, 
    float* ogParticleYValues, 
    float* newParticleYValues, 
    float* ogOldParticleXValues, 
    float* newOldParticleXValues, 
    float* ogOldParticleYValues, 
    float* newOldParticleYValues, 
    int numParticles)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_id < numParticles){
        unsigned short permutedIndex = permutedIndices[thread_id];
        newParticleYValues[thread_id] = ogParticleYValues[permutedIndex];
        newOldParticleXValues[thread_id] = ogOldParticleXValues[permutedIndex];
        newOldParticleYValues[thread_id] = ogOldParticleYValues[permutedIndex];
    }
}


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

    /* TODO: interesting optimization
    sortCounter = (int)((BLOCK_DELTA-SMOOTH)/(2*VEL_LIMIT*PIXEL_PER_METER*dt));
    */

    // First get the current cuda device
    cudaError_t err;
    int dev = 0;
    CUDA_THROW_FAILED(cudaGetDevice(&dev));

    // Check whether grid sync is possible by querying the device attribute
    // TODO
    //int supportsCoopLaunch = 0;
    //CUDA_THROW_FAILED(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    //if(supportsCoopLaunch!=1){
    //    throw std::exception("Cooperative Launch is not supported on the GPU, which is neccesary for the program");
    //}

    // Check the device properties to calculate the maximum amount of allowed particles
    // TODO
    //sharedMemorySize = sizeof(SharedMem);
    //cudaDeviceProp prop;
    //CUDA_THROW_FAILED(cudaGetDeviceProperties(&prop, dev));
    //int maxNumberOfBlocksPerSm = ((prop.sharedMemPerMultiprocessor-1000)/sharedMemorySize); // spare 1000 bytes of shared memory per block just to be safe
    //int maxNumberOfParticlesPerSm = maxNumberOfBlocksPerSm*BLOCK_SIZE;
    //int maxNumberOfParticles = maxNumberOfParticlesPerSm*prop.multiProcessorCount;

    // TODO
    //if(numParticles > maxNumberOfParticles){
    //    throw std::exception("Too many particles are being used, GPU shared memory constraints don't allow usage of this many particles.");
    //}

    //TODO: also check for MAX_BOUNDARIES and MAX_PUMPS

    // Allocate memory on the GPU necessary for the kernel
    allocateDeviceMemory(manager);

    // Transfer the information from the entitymanager to the GPU
    transferToDeviceMemory(manager);
}

PhysicsSystem::~PhysicsSystem() noexcept{
    destroyDeviceMemory();
}

void PhysicsSystem::update(EntityManager& manager){
    if(numParticles==0){
        return;
    }
    
    cudaError_t err;

    /* TODO: DURATION TESTING
    auto start = std::chrono::high_resolution_clock::now();
    */
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<UPDATES_PER_RENDER; i++){
        updateRegularPhysics<<<(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dt, particleXValues[currentParticlesIndex], particleYValues[currentParticlesIndex], oldParticleXValues[currentParticlesIndex], oldParticleYValues[currentParticlesIndex], numParticles, numBoundaries, numPumps);
        /* TODO: interesting optimization
        sortCounter += 1;
        if(2*PIXEL_PER_METER*VEL_LIMIT*dt*sortCounter>=(BLOCK_DELTA-SMOOTH)){
            sortCounter = 0;
            thrust::sort_by_key(thrust::device, particleXValues, particleXValues+numParticles, thrust::make_zip_iterator(thrust::make_tuple(particleYValues, oldParticleXValues, oldParticleYValues)));
        }
        */
        cub::DeviceRadixSort::SortPairs<float, unsigned short>(sortingTempStorage, sortingTempStorageBytes,
            particleXValues[currentParticlesIndex], particleXValues[1-currentParticlesIndex], staticIndexes, permutedIndexes, numParticles);
        permuteArrays<<<(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(permutedIndexes, 
            particleYValues[currentParticlesIndex], 
            particleYValues[1-currentParticlesIndex], 
            oldParticleXValues[currentParticlesIndex], 
            oldParticleXValues[1-currentParticlesIndex],
            oldParticleYValues[currentParticlesIndex],
            oldParticleYValues[1-currentParticlesIndex], 
            numParticles);
        currentParticlesIndex = 1-currentParticlesIndex;
        //thrust::sort_by_key(thrust::device, particleXValues, particleXValues+numParticles, thrust::make_zip_iterator(thrust::make_tuple(particleYValues, oldParticleXValues, oldParticleYValues)));
        updateDensityField<<<(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE,sizeof(SharedMem1)>>>(dt, 
            particleXValues[currentParticlesIndex],
            particleYValues[currentParticlesIndex],
            particlePressureDensityRatios,
            numParticles, 
            numBoundaries,
            numNearbyBoundaries,
            nearbyBoundaryIndices,
            numNearbyParticles,
            nearbyParticleIndices,
            minBlockIterator,
            maxBlockIterator);
        updateParticlesByDensityField<<<(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE,sizeof(SharedMem1)>>>(dt, 
            particleXValues[currentParticlesIndex], 
            particleYValues[currentParticlesIndex], 
            particlePressureDensityRatios,
            numParticles,
            numNearbyBoundaries,
            nearbyBoundaryIndices,
            numNearbyParticles,
            nearbyParticleIndices,
            minBlockIterator,
            maxBlockIterator);
        /* TODO: kernel testing
        int someMaxHost[1] = {0};
        cudaMemcpyFromSymbol(someMaxHost, someMax, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if(someMaxHost[0]!=-1){
            std::string exceptionString = "Some max: "+std::to_string(someMaxHost[0]);
            throw std::exception(exceptionString.c_str());
        }
        */
    }
    Particle* graphicsParticles = (Particle*)manager.getParticles().getMappedAccess();
    copyParticlesToGraphics<<<(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(graphicsParticles, particleXValues[currentParticlesIndex], particleYValues[currentParticlesIndex], numParticles);
    CUDA_THROW_FAILED(cudaGetLastError());
    CUDA_THROW_FAILED(cudaDeviceSynchronize());
    manager.getParticles().unMap();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::string exceptionString = "Timing in microseconds: "+std::to_string(duration.count());
    throw std::exception(exceptionString.c_str());
    /*  TODO: DURATION TESTING
    //CUDA_THROW_FAILED(cudaGetLastError());
    //CUDA_THROW_FAILED(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::string exceptionString = "Timing in microseconds: "+std::to_string(duration.count());
    throw std::exception(exceptionString.c_str());
    */ 
}

void PhysicsSystem::allocateDeviceMemory(EntityManager& manager){
    cudaError_t err;

    if(numParticles > 0){
        CUDA_THROW_FAILED(cudaMalloc((void**)&particleXValues[0], sizeof(float)*numParticles));
        CUDA_THROW_FAILED(cudaMalloc((void**)&particleYValues[0], sizeof(float)*numParticles));
        CUDA_THROW_FAILED(cudaMalloc((void**)&particleXValues[1], sizeof(float)*numParticles));
        CUDA_THROW_FAILED(cudaMalloc((void**)&particleYValues[1], sizeof(float)*numParticles));

        CUDA_THROW_FAILED(cudaMalloc((void**)&oldParticleXValues[0], sizeof(float)*numParticles));
        CUDA_THROW_FAILED(cudaMalloc((void**)&oldParticleYValues[0], sizeof(float)*numParticles));
        CUDA_THROW_FAILED(cudaMalloc((void**)&oldParticleXValues[1], sizeof(float)*numParticles));
        CUDA_THROW_FAILED(cudaMalloc((void**)&oldParticleYValues[1], sizeof(float)*numParticles));

        CUDA_THROW_FAILED(cudaMalloc((void**)&particlePressureDensityRatios, sizeof(float)*numParticles));

        CUDA_THROW_FAILED(cudaMalloc((void**)&numNearbyBoundaries, sizeof(unsigned char)*numParticles));
        CUDA_THROW_FAILED(cudaMalloc((void**)&numNearbyParticles, sizeof(unsigned char)*numParticles));

        CUDA_THROW_FAILED(cudaMalloc((void**)&nearbyBoundaryIndices, sizeof(unsigned char)*MAX_NEARBY_BOUNDARIES*BLOCK_SIZE*((numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE)));
        CUDA_THROW_FAILED(cudaMalloc((void**)&nearbyParticleIndices, sizeof(unsigned short)*MAX_NEARBY_PARTICLES*BLOCK_SIZE*((numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE)));

        CUDA_THROW_FAILED(cudaMalloc((void**)&minBlockIterator, sizeof(int)*((numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE)));
        CUDA_THROW_FAILED(cudaMalloc((void**)&maxBlockIterator, sizeof(int)*((numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE)));

        CUDA_THROW_FAILED(cudaMalloc((void**)&staticIndexes, sizeof(unsigned short)*(numParticles)));
        CUDA_THROW_FAILED(cudaMalloc((void**)&permutedIndexes, sizeof(unsigned short)*(numParticles)));

        cub::DeviceRadixSort::SortPairs(sortingTempStorage, sortingTempStorageBytes,
            particleXValues[0], particleXValues[1], staticIndexes, permutedIndexes, numParticles);
        CUDA_THROW_FAILED(cudaMalloc((void**)&sortingTempStorage, sortingTempStorageBytes));
    }
}

void PhysicsSystem::transferToDeviceMemory(EntityManager& manager){
    cudaError_t err;

    if(numBoundaries > 0){
        CUDA_THROW_FAILED(cudaMemcpyToSymbol(boundaries, manager.getBoundaries().data(), sizeof(Boundary)*numBoundaries));
    }

    if(numParticles > 0){
        Particle* graphicsParticles = (Particle*)manager.getParticles().getMappedAccess();
        initializeParticles<<<(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(graphicsParticles, particleXValues[0], particleYValues[0], oldParticleXValues[0], oldParticleYValues[0], numParticles, staticIndexes);
        CUDA_THROW_FAILED(cudaGetLastError());
        CUDA_THROW_FAILED(cudaDeviceSynchronize());
        manager.getParticles().unMap();
    }

    if(numPumps > 0){
        CUDA_THROW_FAILED(cudaMemcpyToSymbol(pumps, manager.getPumps().data(), sizeof(Pump)*numPumps));
        CUDA_THROW_FAILED(cudaMemcpyToSymbol(pumpVelocities, manager.getPumpVelocities().data(), sizeof(PumpVelocity)*numPumps));
    }
}

void PhysicsSystem::destroyDeviceMemory() noexcept{
    if(particleXValues){
        cudaFree(particleXValues);
    }

    if(particleYValues){
        cudaFree(particleYValues);
    }

    if(oldParticleXValues){
        cudaFree(oldParticleXValues);
    }

    if(oldParticleYValues){
        cudaFree(oldParticleYValues);
    }

    if(particlePressureDensityRatios){
        cudaFree(particlePressureDensityRatios);
    }

    if(numNearbyBoundaries){
        cudaFree(numNearbyBoundaries);
    }

    if(nearbyBoundaryIndices){
        cudaFree(nearbyBoundaryIndices);
    }

    if(numNearbyParticles){
        cudaFree(numNearbyParticles);
    }

    if(nearbyParticleIndices){
        cudaFree(nearbyParticleIndices);
    }

    if(minBlockIterator){
        cudaFree(minBlockIterator);
    }

    if(maxBlockIterator){
        cudaFree(maxBlockIterator);
    }
}

__global__ void initializeParticles(Particle* graphicsParticles, float* particleXValues, float* particleYValues, float* oldParticleXValues, float* oldParticleYValues, int numParticles, unsigned short* staticIndexes){
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_id < numParticles){ 
        particleXValues[thread_id] = graphicsParticles[thread_id].x;
        particleYValues[thread_id] = graphicsParticles[thread_id].y;

        oldParticleXValues[thread_id] = graphicsParticles[thread_id].x;
        oldParticleYValues[thread_id] = graphicsParticles[thread_id].y;

        staticIndexes[thread_id] = thread_id;
    }
}

__global__ void copyParticlesToGraphics(Particle* graphicsParticles, float* particleXValues, float* particleYValues, int numParticles){
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_id < numParticles){ 
        __stcg((float*)(graphicsParticles+thread_id), __ldcg(particleXValues+thread_id));
        __stcg((float*)(graphicsParticles+thread_id)+1, __ldcg(particleYValues+thread_id));
    }
}

__global__ void updateRegularPhysics(float dt, float* particleXValues, float* particleYValues, float* oldParticleXValues, float* oldParticleYValues, int numParticles, int numBoundaries, int numPumps){
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_id < numParticles){
        float old_x = oldParticleXValues[thread_id];
        float old_y = oldParticleYValues[thread_id];
        float my_x = particleXValues[thread_id];
        float my_y = particleYValues[thread_id];

        float vel_x = (my_x - old_x) / (PIXEL_PER_METER*dt);
        float vel_y = (my_y - old_y) / (PIXEL_PER_METER*dt);

        // Update velocities based on whether the particle is in a pump or not
        for(int i = 0; i < numPumps; i++){
            if(my_x >= pumps[i].xLow && my_x <= pumps[i].xHigh && my_y >= pumps[i].yLow && my_y <= pumps[i].yHigh){
                vel_x = pumpVelocities[i].velX;
                vel_y = pumpVelocities[i].velY;
                break;
            }
        }

        // Update positional change of particles caused by gravity
        vel_y -= GRAVITY*dt;

        // Update particle positions
        my_x += PIXEL_PER_METER*vel_x*dt;
        my_y += PIXEL_PER_METER*vel_y*dt;


        // Update positional change of particles caused by boundaries (make sure particles cannot pass boundaries)
        for(int i=0; i<numBoundaries; i++){
            Boundary line = boundaries[i];
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
        oldParticleXValues[thread_id] = my_x-PIXEL_PER_METER*vel_x*dt;
        oldParticleYValues[thread_id] = my_y-PIXEL_PER_METER*vel_y*dt;

        // Store the new particle positions
        particleXValues[thread_id] = my_x;
        particleYValues[thread_id] = my_y;
    }
}

__forceinline__ __device__ void iterateSharedMemParticles(
    int thread_id,
    SharedMem1* sharedMemPtr,
    float my_x,
    float my_y,
    float& dens,
    unsigned char numNearbyBoundariesReg,
    unsigned char& numNearbyParticlesReg,
    int blockIterator,
    int numParticles,
    unsigned short* nearbyParticleIndices,
    float& minXValue,
    float& maxXValue)
{
    minXValue = FLT_MAX;
    maxXValue = -FLT_MAX;
    for(int warpIterator=-1; warpIterator!=warpId; warpIterator = ((warpIterator+1) % (blockDim.x/WARP_SIZE))){
        if(warpIterator==-1){
            warpIterator = warpId;
        }
        float particleX = sharedMemPtr->particleXValues[laneId+WARP_SIZE*warpIterator];
        float particleY = sharedMemPtr->particleYValues[laneId+WARP_SIZE*warpIterator];

        for(int laneIterator=-1; laneIterator!=laneId; laneIterator = ((laneIterator+1) % (WARP_SIZE))){
            if(laneIterator==-1){
                laneIterator=laneId;
            }
            else{
                particleX = __shfl_sync(0xffffffff, particleX, ((laneId+1) % WARP_SIZE));
                particleY = __shfl_sync(0xffffffff, particleY, ((laneId+1) % WARP_SIZE));
            }
            if(blockIterator*blockDim.x+warpIterator*WARP_SIZE+laneIterator<numParticles){
                minXValue = min(particleX, minXValue);
                maxXValue = max(particleX, maxXValue);
                if(thread_id<numParticles){
                    float dist_squared = (my_x-particleX)*(my_x-particleX)+(my_y-particleY)*(my_y-particleY);
                    if (dist_squared < SMOOTH*SMOOTH) {
                        // If the other particle is close enough, iterate over the closeby boundaries to achieve two things:
                        //  - Convert this particle to a ghost particle over the boundary
                        //  - Check whether the connection between this particle and the particle of the thread crosses a boundary
                        //    in which case the particle is not actually a true neighbour
                        float accumulated_ghost_particle_density = 0.0;
                        int j=0;
                        for(; j<numNearbyBoundariesReg; j++){
                            Boundary line = boundaries[sharedMemPtr->nearbyBoundaryIndices[BLOCK_SIZE*j+threadIdx.x]];
                            float line_nx = line.y2-line.y1;
                            float line_ny = line.x1-line.x2;
                            
                            // Check whether particle crosses this boundary
                            {
                                float first_check = ((my_x-line.x1)*line_nx+(my_y-line.y1)*line_ny)*((particleX-line.x1)*line_nx+(particleY-line.y1)*line_ny);
                                if(first_check <= 0.0){
                                    float second_check1 = (line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny;
                                    float second_check2 = (particleX-my_x)*line_nx+(particleY-my_y)*line_ny;
                                    float crossing_x = my_x;
                                    float crossing_y = my_y;
                                    if(second_check2 > 0.0){
                                        crossing_x += (particleX-my_x)*second_check1/second_check2;
                                        crossing_y += (particleY-my_y)*second_check1/second_check2;
                                    }
                                    float second_check3 = (crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1);
                                    float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
                                    if(second_check3<=(line.x2-line.x1)*(line.x2-line.x1)+(line.y2-line.y1)*(line.y2-line.y1) && second_check4>=0.0) break;
                                }
                            }

                            // Create a ghost particle over the boundary corresponding to this neighbour
                            {
                                float projection = (line.x1-particleX)*line_nx +(line.y1-particleY)*line_ny;
                                float virtual_x = particleX + 2*projection*line_nx/(line_nx*line_nx+line_ny*line_ny);
                                float virtual_y = particleY + 2*projection*line_ny/(line_nx*line_nx+line_ny*line_ny);
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

                        if(j<numNearbyBoundariesReg) continue;
                        
                        // Change the density caused by ghost particles
                        dens += accumulated_ghost_particle_density;

                        if(blockIterator*blockDim.x+warpIterator*WARP_SIZE+laneIterator==thread_id) continue;

                        // Change the density because of the neighbour particle and also add the particle to the neighbours list
                        float q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist_squared / (SMOOTH*SMOOTH/4)));
                        dens += M_P*q2;
                        nearbyParticleIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_PARTICLES+numNearbyParticlesReg)+threadIdx.x] = blockIterator*blockDim.x+warpIterator*WARP_SIZE+laneIterator;
                        numNearbyParticlesReg++;
                    }
                }
            }
        }
    }
}

__global__ void updateDensityField(float dt, 
    float* particleXValues, 
    float* particleYValues, 
    float* particlePressureDensityRatios,
    int numParticles, 
    int numBoundaries,
    unsigned char* numNearbyBoundaries,
    unsigned char* nearbyBoundaryIndices,
    unsigned char* numNearbyParticles,
    unsigned short* nearbyParticleIndices,
    int* minBlockIterator,
    int* maxBlockIterator)
{
    // Initialize shared memory pointers
    extern __shared__ SharedMem1 sharedMemPtr[];

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned char numNearbyBoundariesReg = 0;
    float my_x = 0.0;
    float my_y = 0.0;

    if(thread_id < numParticles){
        my_x = particleXValues[thread_id];
        my_y = particleYValues[thread_id];

        // Look for boundaries near the particle
        for(unsigned char i=0; i<numBoundaries; i++){
            Boundary line = boundaries[i];
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
                    // Particle is hovering above this boundary and distance from boundary is less than 
                    sharedMemPtr->nearbyBoundaryIndices[BLOCK_SIZE*numNearbyBoundariesReg+threadIdx.x] = i;
                    nearbyBoundaryIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_BOUNDARIES+numNearbyBoundariesReg)+threadIdx.x] = i;
                    numNearbyBoundariesReg++;
                }
                else if(((line.x1-my_x)*(line.x1-my_x) + (line.y1-my_y)*(line.y1-my_y)) < SMOOTH*SMOOTH && ((line.x1-my_x)*line_nx+(line.y1-my_y)*line_ny) > 0){
                    // Particle is close enough to one endpoint of the boundary
                    sharedMemPtr->nearbyBoundaryIndices[BLOCK_SIZE*numNearbyBoundariesReg+threadIdx.x] = i;
                    nearbyBoundaryIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_BOUNDARIES+numNearbyBoundariesReg)+threadIdx.x] = i;
                    numNearbyBoundariesReg++;
                }
                else if(((line.x2-my_x)*(line.x2-my_x) + (line.y2-my_y)*(line.y2-my_y)) < SMOOTH*SMOOTH && ((line.x2-my_x)*line_nx+(line.y2-my_y)*line_ny) > 0){
                    // Particle is close enough to another endpoint of the boundary
                    sharedMemPtr->nearbyBoundaryIndices[BLOCK_SIZE*numNearbyBoundariesReg+threadIdx.x] = i;
                    nearbyBoundaryIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_BOUNDARIES+numNearbyBoundariesReg)+threadIdx.x] = i;
                    numNearbyBoundariesReg++;
                }
            }
        }

        numNearbyBoundaries[thread_id] = numNearbyBoundariesReg;
    }

    float q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( 0 / (SMOOTH*SMOOTH/4)));
    float dens = M_P*q2;
    sharedMemPtr->particleXValues[threadIdx.x] = my_x;
    sharedMemPtr->particleYValues[threadIdx.x] = my_y;

    unsigned char numNearbyParticlesReg = 0;
    float minXValueTB;
    float maxXValueTB;

    __syncthreads();

    iterateSharedMemParticles(
        thread_id,
        sharedMemPtr,
        my_x,
        my_y,
        dens,
        numNearbyBoundariesReg,
        numNearbyParticlesReg,
        blockIdx.x,
        numParticles,
        nearbyParticleIndices,
        minXValueTB,
        maxXValueTB);

    __syncthreads();

    int blockIterator=blockIdx.x+1;
    for(; blockIterator<gridDim.x; blockIterator = (blockIterator+1)){
        float minXValue;
        float maxXValue;
        float particleX = 0.0;
        float particleY = 0.0;

        if(threadIdx.x+blockIterator*blockDim.x<numParticles){
            particleX = particleXValues[threadIdx.x+blockIterator*blockDim.x];
            particleY = particleYValues[threadIdx.x+blockIterator*blockDim.x];
        }

        sharedMemPtr->particleXValues[threadIdx.x] = particleX;
        sharedMemPtr->particleYValues[threadIdx.x] = particleY;

        __syncthreads();

        iterateSharedMemParticles(
            thread_id,
            sharedMemPtr,
            my_x,
            my_y,
            dens,
            numNearbyBoundariesReg,
            numNearbyParticlesReg,
            blockIterator,
            numParticles,
            nearbyParticleIndices,
            minXValue,
            maxXValue);
        
        __syncthreads();
        
        if(maxXValue>(maxXValueTB+BLOCK_DELTA)){
            blockIterator++;
            break;
        }
    }
    if(threadIdx.x==0){
        maxBlockIterator[blockIdx.x] = blockIterator;
    }
    
    blockIterator=blockIdx.x-1;
    for(; blockIterator>=0; blockIterator = (blockIterator-1)){
        float minXValue;
        float maxXValue;
        float particleX = 0.0;
        float particleY = 0.0;

        if(threadIdx.x+blockIterator*blockDim.x<numParticles){
            particleX = particleXValues[threadIdx.x+blockIterator*blockDim.x];
            particleY = particleYValues[threadIdx.x+blockIterator*blockDim.x];
        }

        sharedMemPtr->particleXValues[threadIdx.x] = particleX;
        sharedMemPtr->particleYValues[threadIdx.x] = particleY;

        __syncthreads();

        iterateSharedMemParticles(
            thread_id,
            sharedMemPtr,
            my_x,
            my_y,
            dens,
            numNearbyBoundariesReg,
            numNearbyParticlesReg,
            blockIterator,
            numParticles,
            nearbyParticleIndices,
            minXValue,
            maxXValue);
        
        __syncthreads();
        
        if(minXValue<(minXValueTB-BLOCK_DELTA)){
            blockIterator--;
            break;
        }
    }
    if(threadIdx.x==0){
        minBlockIterator[blockIdx.x] = blockIterator;
    }

    if(thread_id<numParticles){
        numNearbyParticles[thread_id] = numNearbyParticlesReg;

        /* TODO: kernel testing
        atomicMax(maxNeighbours, numNearbyParticlesReg);
        */
        
        if(dens>0.0){
            // Calculate the pressure_density_ratio
            particlePressureDensityRatios[thread_id] = STIFF*(dens-REST)/(dens*dens);
        }
    }
}

__forceinline__ __device__ void addGhostBoundaryParticles(float dt, 
    int thread_id,
    SharedMem1* sharedMemPtr,
    float my_x,
    float my_y,
    float particleX,
    float particleY,
    float press,
    unsigned char numNearbyBoundariesReg,
    float& boundaries_vel_x,
    float& boundaries_vel_y)
{
    for(int j=0; j<numNearbyBoundariesReg; j++){
        Boundary line = boundaries[sharedMemPtr->nearbyBoundaryIndices[BLOCK_SIZE*j+threadIdx.x]];
        float line_nx = (line.y2-line.y1);
        float line_ny = (line.x1-line.x2);
        float projection = (line.x1-particleX)*line_nx +(line.y1-particleY)*line_ny;
        float virtual_x = particleX + 2*projection*line_nx/(line_nx*line_nx+line_ny*line_ny);
        float virtual_y = particleY + 2*projection*line_ny/(line_nx*line_nx+line_ny*line_ny);
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

__global__ void updateParticlesByDensityField(float dt, 
    float* particleXValues, 
    float* particleYValues, 
    float* particlePressureDensityRatios,
    int numParticles,
    unsigned char* numNearbyBoundaries,
    unsigned char* nearbyBoundaryIndices,
    unsigned char* numNearbyParticles,
    unsigned short* nearbyParticleIndices,
    int* minBlockIterator,
    int* maxBlockIterator)
{
    // Initialize shared memory pointers
    extern __shared__ SharedMem1 sharedMemPtr[];

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned char numNearbyBoundariesReg = 0;
    unsigned char numNearbyParticlesReg = 0;

    if(thread_id<numParticles){
        numNearbyBoundariesReg = numNearbyBoundaries[thread_id];
        numNearbyParticlesReg = numNearbyParticles[thread_id];
    }

    for(int i=0; i<numNearbyBoundariesReg; i++){
        sharedMemPtr->nearbyBoundaryIndices[BLOCK_SIZE*i+threadIdx.x] = nearbyBoundaryIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_BOUNDARIES+i)+threadIdx.x];
    }

    float my_x = 0.0;
    float my_y = 0.0;
    float myPressureDensityRatio = 0.0;

    float vel_x = 0.0;
    float vel_y = 0.0;
    float boundaries_vel_x = 0.0;
    float boundaries_vel_y = 0.0;

    if(thread_id<numParticles){
        my_x = particleXValues[thread_id];
        my_y = particleYValues[thread_id];
        myPressureDensityRatio = particlePressureDensityRatios[thread_id];

        addGhostBoundaryParticles(dt, 
            thread_id,
            sharedMemPtr,
            my_x,
            my_y,
            my_x,
            my_y,
            M_P*2.0*myPressureDensityRatio,
            numNearbyBoundariesReg,
            boundaries_vel_x,
            boundaries_vel_y);
    }

    sharedMemPtr->particleXValues[threadIdx.x] = my_x;
    sharedMemPtr->particleYValues[threadIdx.x] = my_y;
    sharedMemPtr->particlePressureDensityRatios[threadIdx.x] = myPressureDensityRatio;
    __syncthreads();

    int nearbyParticlesIterator = 0;
    unsigned short nextNearbyParticleIndex = (numNearbyParticlesReg>nearbyParticlesIterator) ? nearbyParticleIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_PARTICLES+nearbyParticlesIterator)+threadIdx.x] : USHRT_MAX;

    while(nextNearbyParticleIndex>=blockIdx.x*BLOCK_SIZE && nextNearbyParticleIndex<(blockIdx.x+1)*BLOCK_SIZE){
        float particleX = sharedMemPtr->particleXValues[nextNearbyParticleIndex-blockIdx.x*BLOCK_SIZE];
        float particleY = sharedMemPtr->particleYValues[nextNearbyParticleIndex-blockIdx.x*BLOCK_SIZE];
        float particlePressureDensityRatio = sharedMemPtr->particlePressureDensityRatios[nextNearbyParticleIndex-blockIdx.x*BLOCK_SIZE];
        float press = M_P*(myPressureDensityRatio + particlePressureDensityRatio);

        // First calculate displacement of the particle caused by neighbours
        {
            float dist_squared = (my_x-particleX)*(my_x-particleX)+(my_y-particleY)*(my_y-particleY);
            float q = (float)(2.0*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
            float displace = (press * q) * dt;
            float abx = (my_x - particleX);
            float aby = (my_y - particleY);
            vel_x += displace * abx;
            vel_y += displace * aby;
        }

        addGhostBoundaryParticles(dt, 
            thread_id,
            sharedMemPtr,
            my_x,
            my_y,
            particleX,
            particleY,
            press,
            numNearbyBoundariesReg,
            boundaries_vel_x,
            boundaries_vel_y);
        
        nearbyParticlesIterator++;
        nextNearbyParticleIndex = (numNearbyParticlesReg>nearbyParticlesIterator) ? nearbyParticleIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_PARTICLES+nearbyParticlesIterator)+threadIdx.x] : USHRT_MAX;
    }

    __syncthreads();

    int maxBlockIteratorReg = maxBlockIterator[blockIdx.x];
    for(int blockIterator=blockIdx.x+1; blockIterator<maxBlockIteratorReg; blockIterator = (blockIterator+1)){
        float particleX = 0.0;
        float particleY = 0.0;
        float particlePressureDensityRatio = 0.0;

        if(threadIdx.x+blockIterator*blockDim.x<numParticles){
            particleX = __ldcg(particleXValues+threadIdx.x+blockIterator*blockDim.x);
            particleY = __ldcg(particleYValues+threadIdx.x+blockIterator*blockDim.x);
            particlePressureDensityRatio = __ldcg(particlePressureDensityRatios+threadIdx.x+blockIterator*blockDim.x);
        }

        sharedMemPtr->particleXValues[threadIdx.x] = particleX;
        sharedMemPtr->particleYValues[threadIdx.x] = particleY;
        sharedMemPtr->particlePressureDensityRatios[threadIdx.x] = particlePressureDensityRatio;
        __syncthreads();

        while(nextNearbyParticleIndex>=blockIterator*BLOCK_SIZE && nextNearbyParticleIndex<(blockIterator+1)*BLOCK_SIZE){
            particleX = sharedMemPtr->particleXValues[nextNearbyParticleIndex-blockIterator*BLOCK_SIZE];
            particleY = sharedMemPtr->particleYValues[nextNearbyParticleIndex-blockIterator*BLOCK_SIZE];
            particlePressureDensityRatio = sharedMemPtr->particlePressureDensityRatios[nextNearbyParticleIndex-blockIterator*BLOCK_SIZE];
            float press = M_P*(myPressureDensityRatio + particlePressureDensityRatio);

            // First calculate displacement of the particle caused by neighbours
            {
                float dist_squared = (my_x-particleX)*(my_x-particleX)+(my_y-particleY)*(my_y-particleY);
                float q = (float)(2.0*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                float displace = (press * q) * dt;
                float abx = (my_x - particleX);
                float aby = (my_y - particleY);
                vel_x += displace * abx;
                vel_y += displace * aby;
            }

            addGhostBoundaryParticles(dt, 
                thread_id,
                sharedMemPtr,
                my_x,
                my_y,
                particleX,
                particleY,
                press,
                numNearbyBoundariesReg,
                boundaries_vel_x,
                boundaries_vel_y);
            
            nearbyParticlesIterator++;
            nextNearbyParticleIndex = (numNearbyParticlesReg>nearbyParticlesIterator) ? nearbyParticleIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_PARTICLES+nearbyParticlesIterator)+threadIdx.x] : USHRT_MAX;
        }

        __syncthreads();
    }

    int minBlockIteratorReg = minBlockIterator[blockIdx.x];
    for(int blockIterator=blockIdx.x-1; blockIterator>minBlockIteratorReg; blockIterator = (blockIterator-1)){
        float particleX = 0.0;
        float particleY = 0.0;
        float particlePressureDensityRatio = 0.0;

        if(threadIdx.x+blockIterator*blockDim.x<numParticles){
            particleX = __ldcg(particleXValues+threadIdx.x+blockIterator*blockDim.x);
            particleY = __ldcg(particleYValues+threadIdx.x+blockIterator*blockDim.x);
            particlePressureDensityRatio = __ldcg(particlePressureDensityRatios+threadIdx.x+blockIterator*blockDim.x);
        }

        sharedMemPtr->particleXValues[threadIdx.x] = particleX;
        sharedMemPtr->particleYValues[threadIdx.x] = particleY;
        sharedMemPtr->particlePressureDensityRatios[threadIdx.x] = particlePressureDensityRatio;
        __syncthreads();

        while(nextNearbyParticleIndex>=blockIterator*BLOCK_SIZE && nextNearbyParticleIndex<(blockIterator+1)*BLOCK_SIZE){
            particleX = sharedMemPtr->particleXValues[nextNearbyParticleIndex-blockIterator*BLOCK_SIZE];
            particleY = sharedMemPtr->particleYValues[nextNearbyParticleIndex-blockIterator*BLOCK_SIZE];
            particlePressureDensityRatio = sharedMemPtr->particlePressureDensityRatios[nextNearbyParticleIndex-blockIterator*BLOCK_SIZE];
            float press = M_P*(myPressureDensityRatio + particlePressureDensityRatio);

            // First calculate displacement of the particle caused by neighbours
            {
                float dist_squared = (my_x-particleX)*(my_x-particleX)+(my_y-particleY)*(my_y-particleY);
                float q = (float)(2.0*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                float displace = (press * q) * dt;
                float abx = (my_x - particleX);
                float aby = (my_y - particleY);
                vel_x += displace * abx;
                vel_y += displace * aby;
            }

            addGhostBoundaryParticles(dt, 
                thread_id,
                sharedMemPtr,
                my_x,
                my_y,
                particleX,
                particleY,
                press,
                numNearbyBoundariesReg,
                boundaries_vel_x,
                boundaries_vel_y);
            
            nearbyParticlesIterator++;
            nextNearbyParticleIndex = (numNearbyParticlesReg>nearbyParticlesIterator) ? nearbyParticleIndices[BLOCK_SIZE*(blockIdx.x*MAX_NEARBY_PARTICLES+nearbyParticlesIterator)+threadIdx.x] : USHRT_MAX;
        }

        __syncthreads();
    }


    /* TODO: kernel testing
    if(nearbyParticlesIterator!=numNearbyParticlesReg){
        atomicMax(someMax, numNearbyParticlesReg-nearbyParticlesIterator);
    }
    */

    if(thread_id < numParticles){
        float boundary_average_nx = 0.0;
        float boundary_average_ny = 0.0;

        //TODO: calculate this in addGhostBoundaryParticles
        for(int i=0; i<numNearbyBoundariesReg; i++){
            Boundary line = boundaries[sharedMemPtr->nearbyBoundaryIndices[BLOCK_SIZE*i+threadIdx.x]];
            float line_nx = line.y2-line.y1;
            float line_ny = line.x1-line.x2;
            float multiplier = rsqrt(line_nx*line_nx+line_ny*line_ny);
            line_nx *= multiplier;
            line_ny *= multiplier;
            boundary_average_nx += line_nx;
            boundary_average_ny += line_ny;
        }

        if(boundary_average_nx*boundaries_vel_x+boundary_average_ny*boundaries_vel_y <=0.0){
            vel_x += boundaries_vel_x;
            vel_y += boundaries_vel_y;
        }

        vel_x += boundaries_vel_x;
        vel_y += boundaries_vel_y;

        // Put a velocity limit on the particles too allow the system to work still somewhat normally 
        // if some unforeseen behaviour would occur
        if(vel_x*vel_x+vel_y*vel_y > VEL_LIMIT*VEL_LIMIT){
            float multiplier = VEL_LIMIT/sqrt(vel_x*vel_x+vel_y*vel_y);
            vel_x *= multiplier;
            vel_y *= multiplier;
        }

        my_x += PIXEL_PER_METER*vel_x*dt;
        my_y += PIXEL_PER_METER*vel_y*dt;

        particleXValues[thread_id] = my_x;
        particleYValues[thread_id] = my_y;
    }
}