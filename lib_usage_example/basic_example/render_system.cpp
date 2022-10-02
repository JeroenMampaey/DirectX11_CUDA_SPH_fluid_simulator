#include "render_system.h"

#define RECTANGLE_WIDTH 50.0f
#define RECTANGLE_HEIGHT 50.0f
#define CIRCLE_RADIUS 25.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx){
    pFilledRectangleDrawer = gfx.createNewGraphicsBoundObject<FilledRectangleDrawer>(1.0f, 0.0f, 1.0f);
    pLineDrawer = gfx.createNewGraphicsBoundObject<LineDrawer>(0.0f, 0.0f, 0.0f);
    pCircleDrawer = gfx.createNewGraphicsBoundObject<FilledCircleDrawer>(0.2f, 0.2f, 1.0f);
    pHollowRectangleDrawer = gfx.createNewGraphicsBoundObject<HollowRectangleDrawer>(0.0f, 0.0f, 0.0f);
    pTextDrawer = gfx.createNewGraphicsBoundObject<DynamicTextDrawer>(1.0f, 1.0f, 1.0f);
    pFrameController = gfx.createNewGraphicsBoundObject<FrameController>(0.6f, 0.8f, 1.0f);
    pViewProjectionController = gfx.createNewGraphicsBoundObject<ViewProjectionController>();
    pViewProjectionController->setProjection(
        DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f)
    );
}

void RenderSystem::update(EntityManager& manager) const{
    pFrameController->beginFrame();
    pViewProjectionController->setView(DirectX::XMMatrixTranslation(-manager.getCamera().x, -manager.getCamera().y, 0.0f));
    for(FilledRectangleEntity& rectangle : manager.getFilledRectangles()){
        pFilledRectangleDrawer->drawFilledRectangle(rectangle.x, rectangle.y, RECTANGLE_WIDTH, RECTANGLE_HEIGHT);
    }
    for(LineEntity& line : manager.getLines()){
        pLineDrawer->drawLine(line.x1, line.y1, line.x2, line.y2);
    }
    for(CircleEntity& circle : manager.getCircles()){
        pCircleDrawer->drawFilledCircle(circle.x, circle.y, CIRCLE_RADIUS);
    }
    for(HollowRectangleEntity& rectangle : manager.getHollowRectangles()){
        pHollowRectangleDrawer->drawHollowRectangle(rectangle.x, rectangle.y, RECTANGLE_WIDTH, RECTANGLE_HEIGHT);
    }
    // Notice: this drawDynamicText call will throw an exception if more than 255 characters are contained in the textfield
    pTextDrawer->drawDynamicText(manager.getSpecificTextField().text, 50.0f, 50.0f, 25.0f, 50.0f);
    pFrameController->drawFrame();
}