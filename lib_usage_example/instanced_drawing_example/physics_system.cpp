#include "physics_system.h"

PhysicsSystem::PhysicsSystem(float dt) noexcept
    :
    dt(dt)
{}

void PhysicsSystem::update(EntityManager& manager) noexcept{
    circle_velocity += 9.81f*dt;
    for(CircleEntity& circle : manager.getCircles()){
        circle.y -= circle_velocity*dt;
    }
}