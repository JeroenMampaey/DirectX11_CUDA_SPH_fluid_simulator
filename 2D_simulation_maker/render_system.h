#pragma once

#include "entity_manager.h"

#define RADIUS 7.5f

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) const;

    private:
        GraphicsEngine& gfx;
        std::unique_ptr<FilledCircleDrawer> particleDrawer;
        std::unique_ptr<LineDrawer> boundaryDrawer;
        std::unique_ptr<LineDrawer> boundaryNormalDrawer;
        std::unique_ptr<HollowRectangleDrawer> pumpDrawer;
        std::unique_ptr<LineDrawer> pumpVelocityDrawer;
};