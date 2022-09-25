#include "render_system.h"

RenderSystem::RenderSystem(GraphicsEngine& gfx){
    FilledCircleDrawerInitializationArgs particleArgs = {0.2f, 0.2f, 1.0f};
    particleDrawer = gfx.createNewDrawer<FilledCircleDrawer>(particleArgs);

    LineDrawerInitializationArgs boundaryArgs = {0.0f, 0.0f, 0.0f};
    boundaryDrawer = gfx.createNewDrawer<LineDrawer>(boundaryArgs);

    LineDrawerInitializationArgs boundaryNormalArgs = {1.0f, 0.0f, 0.0f};
    boundaryNormalDrawer = gfx.createNewDrawer<LineDrawer>(boundaryNormalArgs);

    HollowRectangleDrawerInitializationArgs pumpArgs = {0.0f, 1.0f, 0.0f};
    pumpDrawer = gfx.createNewDrawer<HollowRectangleDrawer>(pumpArgs);

    LineDrawerInitializationArgs pumpVelocityArgs = {1.0f, 0.5f, 0.0f};
    pumpVelocityDrawer = gfx.createNewDrawer<LineDrawer>(pumpVelocityArgs);

    gfx.setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
}

void RenderSystem::update(GraphicsEngine& gfx, EntityManager& manager) const{
    gfx.beginFrame(0.6f, 0.8f, 1.0f);
    for(Particle& particle : manager.getParticles()){
        particleDrawer->drawFilledCircle(particle.x, particle.y, RADIUS);
    }
    for(Boundary& boundary : manager.getBoundaries()){
        boundaryDrawer->drawLine(boundary.x1, boundary.y1, boundary.x2, boundary.y2);

        float length = sqrt((boundary.x2-boundary.x1)*(boundary.x2-boundary.x1)+(boundary.y2-boundary.y1)*(boundary.y2-boundary.y1));
        float nx = (boundary.y2-boundary.y1)/length;
        float ny = (boundary.x1-boundary.x2)/length;
        boundaryNormalDrawer->drawLine(boundary.x1+(boundary.x2-boundary.x1)/2.0f, boundary.y1+(boundary.y2-boundary.y1)/2.0f, boundary.x1+(boundary.x2-boundary.x1)/2.0f+nx*30.0f, boundary.y1+(boundary.y2-boundary.y1)/2.0f+ny*30.0f);
    }
    for(Pump& pump : manager.getPumps()){
        pumpDrawer->drawHollowRectangle(pump.x1+(pump.x2-pump.x1)/2.0f, pump.y1+(pump.y2-pump.y1)/2.0f, pump.x2-pump.x1, pump.y2-pump.y1);
        pumpVelocityDrawer->drawLine(pump.x1+(pump.x2-pump.x1)/2.0f, pump.y1+(pump.y2-pump.y1)/2.0f, pump.x1+(pump.x2-pump.x1)/2+pump.vel_x, pump.y1+(pump.y2-pump.y1)/2+pump.vel_y);
    }
    gfx.endFrame();
}