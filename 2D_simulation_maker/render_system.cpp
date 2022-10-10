#include "render_system.h"

#define PUMP_VELOCITY_SCALER 100.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx){
    pFrameController = gfx.createNewGraphicsBoundObject<FrameController>(0.6f, 0.8f, 1.0f);
    pParticleDrawer = gfx.createNewGraphicsBoundObject<FilledCircleDrawer>(0.2f, 0.2f, 1.0f);
    pBoundaryDrawer = gfx.createNewGraphicsBoundObject<LineDrawer>(0.0f, 0.0f, 0.0f);
    pBoundaryNormalDrawer = gfx.createNewGraphicsBoundObject<LineDrawer>(1.0f, 0.0f, 0.0f);
    pPumpDrawer = gfx.createNewGraphicsBoundObject<HollowRectangleDrawer>(0.0f, 1.0f, 0.0f);
    pPumpVelocityDrawer = gfx.createNewGraphicsBoundObject<LineDrawer>(1.0f, 0.5f, 0.0f);

    std::unique_ptr<ViewProjectionController> pViewProjectionController = gfx.createNewGraphicsBoundObject<ViewProjectionController>();
    pViewProjectionController->setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
}

void RenderSystem::update(EntityManager& manager) const{
    pFrameController->beginFrame();
    for(ParticleZone& particleZone : manager.getParticleZones()){
        pParticleDrawer->drawFilledCircle(particleZone.x, particleZone.y, PARTICLE_ZONE_RADIUS);
    }
    for(Boundary& boundary : manager.getBoundaries()){
        pBoundaryDrawer->drawLine(boundary.x1, boundary.y1, boundary.x2, boundary.y2);

        float length = sqrt((boundary.x2-boundary.x1)*(boundary.x2-boundary.x1)+(boundary.y2-boundary.y1)*(boundary.y2-boundary.y1));
        float nx = (boundary.y2-boundary.y1)/length;
        float ny = (boundary.x1-boundary.x2)/length;
        pBoundaryNormalDrawer->drawLine(boundary.x1+(boundary.x2-boundary.x1)/2.0f, boundary.y1+(boundary.y2-boundary.y1)/2.0f, boundary.x1+(boundary.x2-boundary.x1)/2.0f+nx*30.0f, boundary.y1+(boundary.y2-boundary.y1)/2.0f+ny*30.0f);
    }
    for(Pump& pump : manager.getPumps()){
        pPumpDrawer->drawHollowRectangle(pump.x1+(pump.x2-pump.x1)/2.0f, pump.y1+(pump.y2-pump.y1)/2.0f, pump.x2-pump.x1, pump.y2-pump.y1);
        pPumpVelocityDrawer->drawLine(pump.x1+(pump.x2-pump.x1)/2.0f, pump.y1+(pump.y2-pump.y1)/2.0f, pump.x1+(pump.x2-pump.x1)/2+PUMP_VELOCITY_SCALER*pump.vel_x, pump.y1+(pump.y2-pump.y1)/2+PUMP_VELOCITY_SCALER*pump.vel_y);
    }
    pFrameController->drawFrame();
}