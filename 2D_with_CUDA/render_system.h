#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) const;

    private:
        std::unique_ptr<FrameController> pFrameController;
        std::unique_ptr<FilledCircleInstanceDrawer> pParticleDrawer;
        std::unique_ptr<LineDrawer> pBoundaryDrawer;
        std::unique_ptr<HollowRectangleDrawer> pPumpDrawer;
};