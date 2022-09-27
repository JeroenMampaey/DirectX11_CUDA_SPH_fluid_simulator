#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(GraphicsEngine& gfx, EntityManager& manager) const;

    private:
        std::shared_ptr<FilledCircleDrawer> particleDrawer;
        std::shared_ptr<LineDrawer> boundaryDrawer;
        std::shared_ptr<HollowRectangleDrawer> pumpDrawer;
};