#pragma once

#include "graphics_engine.h"

class FrameControllerHelper;

class FrameController : public GraphicsBoundObject<FrameControllerHelper>{
    public:
        LIBRARY_API ~FrameController() noexcept;
        LIBRARY_API void beginFrame() const;
        LIBRARY_API void drawFrame() const;

#ifndef READ_FROM_LIB_HEADER
        FrameController(std::shared_ptr<FrameControllerHelper> pFrameControllerHelper, float red, float green, float blue) noexcept;
#endif

        float red;
        float green;
        float blue;
};