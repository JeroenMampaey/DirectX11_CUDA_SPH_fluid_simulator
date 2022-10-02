#include "render_system.h"

#define RECTANGLE_WIDTH 50.0f
#define RECTANGLE_HEIGHT 50.0f
#define CIRCLE_RADIUS 25.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx1, GraphicsEngine& gfx2){
    std::unique_ptr<ViewProjectionController> pViewProjectionController1 = gfx1.createNewGraphicsBoundObject<ViewProjectionController>();
    pViewProjectionController1->setProjection(
        DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f)
    );

    std::unique_ptr<ViewProjectionController> pViewProjectionController2 = gfx2.createNewGraphicsBoundObject<ViewProjectionController>();
    pViewProjectionController2->setProjection(
        DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f)
    );

    pFrameController1 = gfx1.createNewGraphicsBoundObject<FrameController>(0.6f, 0.8f, 1.0f);
    pFrameController2 = gfx2.createNewGraphicsBoundObject<FrameController>(1.0f, 0.8f, 0.6f);

    pFilledRectangleDrawer = gfx1.createNewGraphicsBoundObject<FilledRectangleDrawer>(1.0f, 0.0f, 1.0f);
    pCircleDrawer = gfx2.createNewGraphicsBoundObject<FilledCircleDrawer>(0.2f, 0.2f, 1.0f);
}

void RenderSystem::update(EntityManager& manager) const{
    pFrameController1->beginFrame();
    for(FilledRectangleEntity& rectangle : manager.getFilledRectangles()){
        pFilledRectangleDrawer->drawFilledRectangle(rectangle.x, rectangle.y, RECTANGLE_WIDTH, RECTANGLE_HEIGHT);
    }
    pFrameController1->drawFrame();

    pFrameController2->beginFrame();
    for(CircleEntity& circle : manager.getCircles()){
        pCircleDrawer->drawFilledCircle(circle.x, circle.y, CIRCLE_RADIUS);
    }
    pFrameController2->drawFrame();
}