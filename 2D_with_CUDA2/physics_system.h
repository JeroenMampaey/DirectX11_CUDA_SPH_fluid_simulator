#pragma once

#include "entity_manager.h"
#include "exceptions.h"

#define UPDATES_PER_RENDER 6

struct CompactParticle{
    float x;
    float y;
};

class PhysicsSystem{
    public:
        PhysicsSystem(GraphicsEngine& gfx, EntityManager& manager);
        ~PhysicsSystem() noexcept;
        void update(EntityManager& manager);
    private:
        void allocateDeviceMemory(EntityManager& manager);
        void transferToDeviceMemory(EntityManager& manager);
        void destroyDeviceMemory() noexcept;

        float dt;
        
        // boundaries is constant memory and thus declared in the .cu file
        int numBoundaries;

        float* particleXValues = nullptr;
        float* particleYValues = nullptr;
        float* oldParticleXValues = nullptr;
        float* oldParticleYValues = nullptr;
        float* particlePressureDensityRatios = nullptr;
        int numParticles;

        // pumps and pumpvelocities is constant memory and thus declared in the .cu file
        int numPumps;

        unsigned char* numNearbyBoundaries = nullptr;
        unsigned char* nearbyBoundaryIndices = nullptr;

        unsigned char* numNearbyParticles = nullptr;
        unsigned short* nearbyParticleIndices = nullptr;

        int* minBlockIterator = nullptr;
        int* maxBlockIterator = nullptr;

        int sharedMemorySize;
};