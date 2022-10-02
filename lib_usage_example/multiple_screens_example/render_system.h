#pragma once

#include "entity_manager.h"

class RenderSystem{
    public:
        RenderSystem(GraphicsEngine& gfx1, GraphicsEngine& gfx2);
        void update(EntityManager& manager) const;

    private:
        std::unique_ptr<FrameController> pFrameController1;
        std::unique_ptr<FrameController> pFrameController2;
        std::unique_ptr<FilledRectangleDrawer> pFilledRectangleDrawer;
        std::unique_ptr<FilledCircleDrawer> pCircleDrawer;
};