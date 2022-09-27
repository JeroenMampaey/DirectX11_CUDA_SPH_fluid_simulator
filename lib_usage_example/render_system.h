#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) const;

    private:
        GraphicsEngine& gfx;
        std::shared_ptr<FilledRectangleDrawer> filledRectangleDrawer;
        std::shared_ptr<LineDrawer> lineDrawer;
        std::shared_ptr<FilledCircleDrawer> circleDrawer;
        std::shared_ptr<HollowRectangleDrawer> hollowRectangleDrawer;
        std::shared_ptr<DynamicTextDrawer> textDrawer;
};