#include "render_system.h"

#define CIRCLE_RADIUS 25.0f

RenderSystem::RenderSystem(GraphicsEngine& gfx, EntityManager& manager){
    pFrameController = gfx.createNewGraphicsBoundObject<FrameController>(0.6f, 0.8f, 1.0f);
    pFilledCircleInstancedDrawer = gfx.createNewGraphicsBoundObject<FilledCircleInstanceDrawer>(0.2f, 0.2f, 1.0f);
    pBuffer = gfx.createNewGraphicsBoundObject<CpuAccessibleFilledCircleInstanceBuffer>(static_cast<int>(manager.getCircles().size()), CIRCLE_RADIUS);
    std::unique_ptr<ViewProjectionController> pViewProjectionController = gfx.createNewGraphicsBoundObject<ViewProjectionController>();
    pViewProjectionController->setProjection(
        DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f)
    );
}

void RenderSystem::update(EntityManager& manager) const{
    pFrameController->beginFrame();

    DirectX::XMFLOAT3* rawBuffer = pBuffer->getMappedAccess();
    for(int i=0; i<pBuffer->numberOfCircles; i++){
        CircleEntity& circle = manager.getCircles()[i];
        rawBuffer[i] = {circle.x, circle.y, 0.0f};
    }
    pBuffer->unMap();
    pFilledCircleInstancedDrawer->drawFilledCircleBuffer(*pBuffer.get());

    pFrameController->drawFrame();
}