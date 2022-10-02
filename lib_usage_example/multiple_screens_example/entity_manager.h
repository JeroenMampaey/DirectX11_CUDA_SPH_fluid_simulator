#pragma once

#include <vector>
#include "../../lib/lib_header.h"

class FilledRectangleEntity{
    public:
        float x;
        float y;
        FilledRectangleEntity(float x, float y) noexcept;
};

class CircleEntity{
    public:
        float x;
        float y;
        CircleEntity(float x, float y) noexcept;
};

class EntityManager{
    public:
        EntityManager() noexcept;
        std::vector<FilledRectangleEntity>& getFilledRectangles() noexcept;
        std::vector<CircleEntity>& getCircles() noexcept;
    
    private:
        std::vector<FilledRectangleEntity> filledRectangles;
        std::vector<CircleEntity> circles;
};