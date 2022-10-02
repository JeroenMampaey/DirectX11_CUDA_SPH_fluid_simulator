#pragma once

#include "entity_manager.h"
#include "exceptions.h"

#define UPDATES_PER_RENDER 6

class PhysicsSystem{
    public:
        PhysicsSystem(GraphicsEngine& gfx, EntityManager& manager);
        ~PhysicsSystem();
        void update(EntityManager& manager);
    private:
        void allocateDeviceMemory(EntityManager& manager);
        void transferToDeviceMemory(EntityManager& manager);
        void destroyDeviceMemory() noexcept;

        float dt;    
        Boundary* boundaries = nullptr;
        int numBoundaries;
        int numParticles;
        Particle* oldParticles = nullptr;
        Pump* pumps = nullptr;
        PumpVelocity* pumpVelocities = nullptr;
        int numPumps;
        float* pressureDensityRatios = nullptr;
};