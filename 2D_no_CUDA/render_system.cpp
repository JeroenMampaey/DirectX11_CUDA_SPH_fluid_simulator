#include "render_system.h"

#define RECTANGLE_WIDTH 50.0f
#define RECTANGLE_HEIGHT 50.0f
#define CIRCLE_RADIUS 25.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx){
    pFrameController = gfx.createNewGraphicsBoundObject<FrameController>(0.6f, 0.8f, 1.0f);
    pParticleDrawer = gfx.createNewGraphicsBoundObject<FilledCircleDrawer>(0.2f, 0.2f, 1.0f);
    pBoundaryDrawer = gfx.createNewGraphicsBoundObject<LineDrawer>(0.0f, 0.0f, 0.0f);
    pPumpDrawer = gfx.createNewGraphicsBoundObject<HollowRectangleDrawer>(0.0f, 1.0f, 0.0f);

    std::unique_ptr<ViewProjectionController> pViewProjectionController = gfx.createNewGraphicsBoundObject<ViewProjectionController>();
    pViewProjectionController->setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
}

void RenderSystem::update(EntityManager& manager) const{
    pFrameController->beginFrame();
    for(Particle& particle : manager.getParticles()){
        pParticleDrawer->drawFilledCircle(particle.pos.x, particle.pos.y, RADIUS);
    }
    for(Boundary& boundary : manager.getBoundaries()){
        pBoundaryDrawer->drawLine(boundary.point1.x, boundary.point1.y, boundary.point2.x, boundary.point2.y);
    }
    for(Pump& pump : manager.getPumps()){
        pPumpDrawer->drawHollowRectangle(pump.leftBottom.x+(pump.rightTop.x-pump.leftBottom.x)/2.0f, pump.leftBottom.y+(pump.rightTop.y-pump.leftBottom.y)/2.0f, pump.rightTop.x-pump.leftBottom.x, pump.rightTop.y-pump.leftBottom.y);
    }
    pFrameController->drawFrame();
}