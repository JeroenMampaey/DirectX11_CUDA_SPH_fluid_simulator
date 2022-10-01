#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx, EntityManager& manager);
        void update(EntityManager& manager) const;

    private:
        GraphicsEngine& gfx;
        std::unique_ptr<FilledCircleInstanceDrawer> filledCircleBufferDrawer;
        std::unique_ptr<FilledRectangleDrawer> filledRectangleDrawer;
        std::unique_ptr<LineDrawer> lineDrawer;
        std::unique_ptr<FilledCircleDrawer> circleDrawer;
        std::unique_ptr<HollowRectangleDrawer> hollowRectangleDrawer;
        std::unique_ptr<DynamicTextDrawer> textDrawer;
};