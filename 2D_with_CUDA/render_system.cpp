#include "render_system.h"

#define RECTANGLE_WIDTH 50.0f
#define RECTANGLE_HEIGHT 50.0f
#define CIRCLE_RADIUS 25.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx)
    :
    gfx(gfx),
    particleDrawer(gfx.createNewDrawer<FilledCircleDrawer>(0.2f, 0.2f, 1.0f)),
    boundaryDrawer(gfx.createNewDrawer<LineDrawer>(0.0f, 0.0f, 0.0f)),
    pumpDrawer(gfx.createNewDrawer<HollowRectangleDrawer>(0.0f, 1.0f, 0.0f))
{
    gfx.setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
}

void RenderSystem::update(EntityManager& manager) const{
    gfx.beginFrame(0.6f, 0.8f, 1.0f);
    for(Particle& particle : manager.getParticles()){
        particleDrawer->drawFilledCircle(particle.x, particle.y, RADIUS);
    }
    for(Boundary& boundary : manager.getBoundaries()){
        boundaryDrawer->drawLine((float)boundary.x1, (float)boundary.y1, (float)boundary.x2, (float)boundary.y2);
    }
    for(Pump& pump : manager.getPumps()){
        pumpDrawer->drawHollowRectangle(((float)pump.xLow)+((float)(pump.xHigh-pump.xLow))/2.0f, ((float)pump.yLow)+((float)(pump.yHigh-pump.yLow))/2.0f, (float)(pump.xHigh-pump.xLow), (float)(pump.yHigh-pump.yLow));
    }
    gfx.endFrame();
}