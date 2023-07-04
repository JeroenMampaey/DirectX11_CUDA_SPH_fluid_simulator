#pragma once

#include "entity_manager.h"

class PhysicsSystem{
    public:
        PhysicsSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) const noexcept;
    private:
        inline void performNonFluidRelatedPhysics(EntityManager& manager) const noexcept;
        inline void updateDensityFieldCausedByNeighbours(EntityManager& manager) const noexcept;
        inline void updateDensityFieldCausedByGhostParticles(EntityManager& manager) const noexcept;
        inline void updatePressureField(EntityManager& manager) const noexcept;
        inline void applyEulersEquation(EntityManager& manager) const noexcept;

        inline void addGhostParticleHelper(Particle& p, const Boundary& line, const Particle& neighbor_particle) const noexcept;

        int updatesPerRender;
        float dt;
};