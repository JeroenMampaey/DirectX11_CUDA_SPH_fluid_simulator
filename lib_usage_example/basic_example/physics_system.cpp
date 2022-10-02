#include "physics_system.h"

PhysicsSystem::PhysicsSystem(float dt) noexcept
    :
    dt(dt)
{}

void PhysicsSystem::update(EntityManager& manager) noexcept{
    rectangle_velocity += 9.81f*dt;
    for(FilledRectangleEntity& rectangle : manager.getFilledRectangles()){
        rectangle.y -= rectangle_velocity*dt;
    }
}