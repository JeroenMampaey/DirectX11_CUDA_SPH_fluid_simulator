#include "entity_manager.h"

CameraEntity::CameraEntity(float x, float y) noexcept
    :
    x(x),
    y(y)
{}

FilledRectangleEntity::FilledRectangleEntity(float x, float y) noexcept
    :
    x(x),
    y(y)
{}

LineEntity::LineEntity(float x1, float y1, float x2, float y2) noexcept
    :
    x1(x1),
    y1(y1),
    x2(x2),
    y2(y2)
{}

CircleEntity::CircleEntity(float x, float y) noexcept
    :
    x(x),
    y(y)
{}

HollowRectangleEntity::HollowRectangleEntity(float x, float y) noexcept
    :
    x(x),
    y(y)
{}

SpecificTextFieldEntity::SpecificTextFieldEntity(std::string text, int counter) noexcept
    :
    text(text),
    counter(counter)
{}

EntityManager::EntityManager() noexcept
    :
    camera(CameraEntity(0.0f, 0.0f)),
    specificTextField(SpecificTextFieldEntity("", 1)),
    circleCollection(nullptr)
{
    for(int i=0; i<20; i++){
        filledRectangles.push_back(FilledRectangleEntity(50.0f+50.0f*i, (float)(HEIGHT/2)));
    }
    for(int i=0; i<10; i++){
        lines.push_back(LineEntity(i*100.0f, (i-2)*(i-2)*10.0f, (i+1)*100.0f, (i-1)*(i-1)*10.0f));
    }
    for(int i=0; i<20; i++){
        circles.push_back(CircleEntity(50.0f+50.0f*i, (float)(HEIGHT/2)+50.0f));
    }
    for(int i=0; i<10; i++){
        hollowRectangles.push_back(HollowRectangleEntity((i+1)*100.0f, (i-1)*(i-1)*10.0f));
    }
}

CameraEntity& EntityManager::getCamera() noexcept{
    return camera;
}

std::vector<FilledRectangleEntity>& EntityManager::getFilledRectangles() noexcept{
    return filledRectangles;
}

std::vector<LineEntity>& EntityManager::getLines() noexcept{
    return lines;
}

std::vector<CircleEntity>& EntityManager::getCircles() noexcept{
    return circles;
}

std::vector<HollowRectangleEntity>& EntityManager::getHollowRectangles() noexcept{
    return hollowRectangles;
}

SpecificTextFieldEntity& EntityManager::getSpecificTextField() noexcept{
    return specificTextField;
}

std::shared_ptr<FilledCircleInstanceBuffer>& EntityManager::getCircleCollection() noexcept{
    return circleCollection;
}