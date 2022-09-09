#pragma once

#include "../lib/lib_header.h"

class ExampleEngine : public GraphicsEngine{
    public:
        ExampleEngine(HWND hWnd, UINT msPerFrame);
    protected:
        void update();
    private:
        SquareFactory squareFactory = SquareFactory();
        std::unique_ptr<Drawable> mySquare;
};