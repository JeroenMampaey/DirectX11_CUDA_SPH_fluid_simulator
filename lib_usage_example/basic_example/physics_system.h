#pragma once

#include "entity_manager.h"

class PhysicsSystem{
    public:
        PhysicsSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) noexcept;
    private:
        float dt;
        float rectangle_velocity = 0.0f;
};