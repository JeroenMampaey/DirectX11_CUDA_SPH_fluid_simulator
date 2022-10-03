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
           
        Boundary* boundaries = nullptr;
        int numBoundaries;

        CompactParticle* oldParticles = nullptr;
        CompactParticle* positionCommunicationMemory = nullptr;
        float* pressureDensityRatioCommunicationMemory = nullptr;
        int numParticles;

        Pump* pumps = nullptr;
        PumpVelocity* pumpVelocities = nullptr;
        int numPumps;
};