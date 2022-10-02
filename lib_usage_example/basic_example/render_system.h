#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx);
        void update(EntityManager& manager) const;

    private:
        std::unique_ptr<FrameController> pFrameController;
        std::unique_ptr<ViewProjectionController> pViewProjectionController;
        std::unique_ptr<FilledRectangleDrawer> pFilledRectangleDrawer;
        std::unique_ptr<LineDrawer> pLineDrawer;
        std::unique_ptr<FilledCircleDrawer> pCircleDrawer;
        std::unique_ptr<HollowRectangleDrawer> pHollowRectangleDrawer;
        std::unique_ptr<DynamicTextDrawer> pTextDrawer;
};