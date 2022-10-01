#include "render_system.h"

#define RECTANGLE_WIDTH 50.0f
#define RECTANGLE_HEIGHT 50.0f
#define CIRCLE_RADIUS 25.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx, EntityManager& manager)
    :
    gfx(gfx),
    filledRectangleDrawer(gfx.createNewDrawer<FilledRectangleDrawer>(1.0f, 0.0f, 1.0f)),
    lineDrawer(gfx.createNewDrawer<LineDrawer>(0.0f, 0.0f, 0.0f)),
    circleDrawer(gfx.createNewDrawer<FilledCircleDrawer>(0.2f, 0.2f, 1.0f)),
    hollowRectangleDrawer(gfx.createNewDrawer<HollowRectangleDrawer>(0.0f, 0.0f, 0.0f)),
    textDrawer(gfx.createNewDrawer<DynamicTextDrawer>(1.0f, 1.0f, 1.0f)),
    filledCircleBufferDrawer(gfx.createNewDrawer<FilledCircleInstanceDrawer>(0.2f, 0.2f, 1.0f))
{
    manager.getCircleCollection() = filledCircleBufferDrawer->createCpuAccessibleBuffer(2, 50.0f);
    DirectX::XMFLOAT3* test = manager.getCircleCollection()->getMappedAccess();
    test[0] = {250.0f, 250.0f, 0.0f};
    test[1] = {450.0f, 450.0f, 0.0f};
    manager.getCircleCollection()->unMap();

    gfx.setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
}

void RenderSystem::update(EntityManager& manager) const{
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
    // Notice: this drawDynamicText call will throw an exception if more than 255 characters are contained in the textfield
    textDrawer->drawDynamicText(manager.getSpecificTextField().text, 50.0f, 50.0f, 25.0f, 50.0f);
    filledCircleBufferDrawer->drawFilledCircleBuffer(manager.getCircleCollection().get());
    gfx.endFrame();
}