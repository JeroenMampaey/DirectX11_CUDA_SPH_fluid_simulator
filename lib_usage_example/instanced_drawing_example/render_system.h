#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx, EntityManager& manager);
        void update(EntityManager& manager) const;

    private:
        std::unique_ptr<FrameController> pFrameController;
        std::unique_ptr<CpuAccessibleFilledCircleInstanceBuffer> pBuffer;
        std::unique_ptr<FilledCircleInstanceDrawer> pFilledCircleInstancedDrawer;
};