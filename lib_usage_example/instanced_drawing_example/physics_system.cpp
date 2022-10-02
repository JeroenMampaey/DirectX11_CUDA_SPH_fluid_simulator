#include "physics_system.h"

PhysicsSystem::PhysicsSystem(GraphicsEngine& gfx){
    if(RATE_IS_INVALID(gfx.getRefreshRate())){
        throw std::exception("Refreshrate could not easily be found programmatically.");
    }
    dt = 1.0f/gfx.getRefreshRate();
}

void PhysicsSystem::update(EntityManager& manager) noexcept{
    circle_velocity += 9.81f*dt;
    for(CircleEntity& circle : manager.getCircles()){
        circle.y -= circle_velocity*dt;
    }
}