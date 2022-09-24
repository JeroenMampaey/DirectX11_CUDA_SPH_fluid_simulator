#include "render_system.h"

#define RECTANGLE_WIDTH 50.0f
#define RECTANGLE_HEIGHT 50.0f
#define CIRCLE_RADIUS 25.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx){
    FilledRectangleDrawerInitializationArgs filledRectangleArgs = {1.0f, 0.0f, 1.0f};
    filledRectangleDrawer = gfx.createNewDrawer<FilledRectangleDrawer>(filledRectangleArgs);

    LineDrawerInitializationArgs lineArgs = {0.0f, 0.0f, 0.0f};
    lineDrawer = gfx.createNewDrawer<LineDrawer>(lineArgs);

    FilledCircleDrawerInitializationArgs circleArgs = {0.2f, 0.2f, 1.0f};
    circleDrawer = gfx.createNewDrawer<FilledCircleDrawer>(circleArgs);

    HollowRectangleDrawerInitializationArgs hollowRectangleArgs = {0.0f, 0.0f, 0.0f};
    hollowRectangleDrawer = gfx.createNewDrawer<HollowRectangleDrawer>(hollowRectangleArgs);

    DynamicTextDrawerInitializationArgs textArgs = {1.0f, 1.0f, 1.0f};
    textDrawer = gfx.createNewDrawer<DynamicTextDrawer>(textArgs);

    gfx.setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
}

void RenderSystem::update(GraphicsEngine& gfx, EntityManager& manager) const{
    gfx.beginFrame(0.6f, 0.8f, 1.0f);
    gfx.setView(DirectX::XMMatrixTranslation(-manager.getCamera().x, -manager.getCamera().y, 0.0f));
    for(FilledRectangleEntity& rectangle : manager.getFilledRectangles()){
        filledRectangleDrawer->drawFilledRectangle(rectangle.x, rectangle.y, RECTANGLE_WIDTH, RECTANGLE_HEIGHT);
    }
    for(LineEntity& line : manager.getLines()){
        lineDrawer->drawLine(line.x1, line.y1, line.x2, line.y2);
    }
    for(CircleEntity& circle : manager.getCircles()){
        circleDrawer->drawFilledCircle(circle.x, circle.y, CIRCLE_RADIUS);
    }
    for(HollowRectangleEntity& rectangle : manager.getHollowRectangles()){
        hollowRectangleDrawer->drawHollowRectangle(rectangle.x, rectangle.y, RECTANGLE_WIDTH, RECTANGLE_HEIGHT);
    }
    // Notice: this drawtext call will throw an exception if more than 255 characters are contained in the textfield
    textDrawer->drawDynamicText(manager.getSpecificTextField().text, 50.0f, 50.0f, 25.0f, 50.0f);
    gfx.endFrame();
}