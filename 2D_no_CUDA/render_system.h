#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) const;

    private:
        std::unique_ptr<FrameController> pFrameController;
        std::shared_ptr<FilledCircleDrawer> pParticleDrawer;
        std::shared_ptr<LineDrawer> pBoundaryDrawer;
        std::shared_ptr<HollowRectangleDrawer> pPumpDrawer;
};