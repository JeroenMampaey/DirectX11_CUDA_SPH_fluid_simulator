#pragma once

#include <vector>
#include "../../lib/lib_header.h"

class CircleEntity{
    public:
        float x;
        float y;
        CircleEntity(float x, float y) noexcept;
};

class EntityManager{
    public:
        EntityManager() noexcept;
        std::vector<CircleEntity>& getCircles() noexcept;
    
    private:
        std::vector<CircleEntity> circles;
};