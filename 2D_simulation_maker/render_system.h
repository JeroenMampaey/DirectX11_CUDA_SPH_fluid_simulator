#pragma once

#include "entity_manager.h"

#define RADIUS 7.5f

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) const;

    private:
        std::unique_ptr<FrameController> pFrameController;
        std::unique_ptr<FilledCircleDrawer> pParticleDrawer;
        std::unique_ptr<LineDrawer> pBoundaryDrawer;
        std::unique_ptr<LineDrawer> pBoundaryNormalDrawer;
        std::unique_ptr<HollowRectangleDrawer> pPumpDrawer;
        std::unique_ptr<LineDrawer> pPumpVelocityDrawer;
};