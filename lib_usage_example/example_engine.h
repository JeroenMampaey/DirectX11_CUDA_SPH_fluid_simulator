#pragma once

#include "../lib/graphics_engine.h"
#include "../lib/drawables/square.h"

class ExampleEngine : public GraphicsEngine{
    public:
        ExampleEngine(HWND hWnd, UINT msPerFrame);
    protected:
        void update();
    private:
        std::unique_ptr<Square> mySquare;
};