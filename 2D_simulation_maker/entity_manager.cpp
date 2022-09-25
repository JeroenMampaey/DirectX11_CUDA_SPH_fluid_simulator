#include "entity_manager.h"

Particle::Particle(float x, float y)
    :
    x(x),
    y(y)
{}

Boundary::Boundary(float x1, float y1, float x2, float y2)
    :
    x1(x1),
    y1(y1),
    x2(x2),
    y2(y2)
{}

Pump::Pump(float x1, float y1, float x2, float y2, float vel_x, float vel_y)
    :
    x1(x1),
    y1(y1),
    x2(x2),
    y2(y2),
    vel_x(vel_x),
    vel_y(vel_y)
{}

Pump::Pump(float x1, float y1, float x2, float y2)
    :
    x1(x1),
    y1(y1),
    x2(x2),
    y2(y2),
    vel_x(DEFAULT_PUMP_VELOCITY),
    vel_y(0.0f)
{}

std::vector<Particle>& EntityManager::getParticles() noexcept{
    return particles;
}

std::vector<Boundary>& EntityManager::getBoundaries() noexcept{
    return boundaries;
}

std::vector<Pump>& EntityManager::getPumps() noexcept{
    return pumps;
}