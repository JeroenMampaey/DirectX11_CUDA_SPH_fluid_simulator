#pragma once

#include <vector>
#include "../../lib/lib_header.h"

class CameraEntity{
    public:
        float x;
        float y;
        CameraEntity(float x, float y) noexcept;
};

class FilledRectangleEntity{
    public:
        float x;
        float y;
        FilledRectangleEntity(float x, float y) noexcept;
};

class LineEntity{
    public:
        float x1;
        float y1;
        float x2;
        float y2;
        LineEntity(float x1, float y1, float x2, float y2) noexcept;
};

class CircleEntity{
    public:
        float x;
        float y;
        CircleEntity(float x, float y) noexcept;
};

class HollowRectangleEntity{
    public:
        float x;
        float y;
        HollowRectangleEntity(float x, float y) noexcept;
};

class SpecificTextFieldEntity{
    public:
        std::string text;
        int counter;
        SpecificTextFieldEntity(std::string text, int counter) noexcept;
};

class EntityManager{
    public:
        EntityManager() noexcept;
        CameraEntity& getCamera() noexcept;
        std::vector<FilledRectangleEntity>& getFilledRectangles() noexcept;
        std::vector<LineEntity>& getLines() noexcept;
        std::vector<CircleEntity>& getCircles() noexcept;
        std::vector<HollowRectangleEntity>& getHollowRectangles() noexcept;
        SpecificTextFieldEntity& getSpecificTextField() noexcept;
    
    private:
        CameraEntity camera;
        std::vector<FilledRectangleEntity> filledRectangles;
        std::vector<LineEntity> lines;
        std::vector<CircleEntity> circles;
        std::vector<HollowRectangleEntity> hollowRectangles;
        SpecificTextFieldEntity specificTextField;
};