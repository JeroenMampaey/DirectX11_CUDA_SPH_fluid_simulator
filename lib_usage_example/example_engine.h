#pragma once

#include "../lib/graphics_engine.h"

class ExampleEngine : public GraphicsEngine{
    public:
        ExampleEngine(HWND hWnd, UINT msPerFrame);
    protected:
        void update();
};