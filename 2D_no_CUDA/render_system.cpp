#include "render_system.h"

#define RECTANGLE_WIDTH 50.0f
#define RECTANGLE_HEIGHT 50.0f
#define CIRCLE_RADIUS 25.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx){
    FilledCircleDrawerInitializationArgs particleArgs = {0.2f, 0.2f, 1.0f};
    particleDrawer = gfx.createNewDrawer<FilledCircleDrawer>(particleArgs);

    LineDrawerInitializationArgs boundaryArgs = {0.0f, 0.0f, 0.0f};
    boundaryDrawer = gfx.createNewDrawer<LineDrawer>(boundaryArgs);

    HollowRectangleDrawerInitializationArgs pumpArgs = {0.0f, 1.0f, 0.0f};
    pumpDrawer = gfx.createNewDrawer<HollowRectangleDrawer>(pumpArgs);

    gfx.setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
}

void RenderSystem::update(GraphicsEngine& gfx, EntityManager& manager) const{
    gfx.beginFrame(0.6f, 0.8f, 1.0f);
    for(Particle& particle : manager.getParticles()){
        particleDrawer->drawFilledCircle(particle.pos.x, particle.pos.y, RADIUS);
    }
    for(Boundary& boundary : manager.getBoundaries()){
        boundaryDrawer->drawLine(boundary.point1.x, boundary.point1.y, boundary.point2.x, boundary.point2.y);
    }
    for(Pump& pump : manager.getPumps()){
        pumpDrawer->drawHollowRectangle(pump.leftBottom.x+(pump.rightTop.x-pump.leftBottom.x)/2.0f, pump.leftBottom.y+(pump.rightTop.y-pump.leftBottom.y)/2.0f, pump.rightTop.x-pump.leftBottom.x, pump.rightTop.y-pump.leftBottom.y);
    }
    gfx.endFrame();
}