#include "entity_manager.h"

CircleEntity::CircleEntity(float x, float y) noexcept
    :
    x(x),
    y(y)
{}

EntityManager::EntityManager() noexcept{
    for(int i=0; i<20; i++){
        circles.push_back(CircleEntity(50.0f+50.0f*i, (float)(HEIGHT/2)+50.0f));
    }
}

std::vector<CircleEntity>& EntityManager::getCircles() noexcept{
    return circles;
}