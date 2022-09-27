#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) const;

    private:
        GraphicsEngine& gfx;
        std::shared_ptr<FilledCircleDrawer> particleDrawer;
        std::shared_ptr<LineDrawer> boundaryDrawer;
        std::shared_ptr<HollowRectangleDrawer> pumpDrawer;
};