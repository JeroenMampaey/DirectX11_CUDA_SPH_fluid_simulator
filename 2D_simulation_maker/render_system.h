#pragma once

#include "entity_manager.h"

#define RADIUS 7.5f

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(GraphicsEngine& gfx, EntityManager& manager) const;

    private:
        std::shared_ptr<FilledCircleDrawer> particleDrawer;
        std::shared_ptr<LineDrawer> boundaryDrawer;
        std::shared_ptr<LineDrawer> boundaryNormalDrawer;
        std::shared_ptr<HollowRectangleDrawer> pumpDrawer;
        std::shared_ptr<LineDrawer> pumpVelocityDrawer;
};