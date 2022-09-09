#pragma once

#include "../lib/lib_header.h"

class ExampleEngine : public GraphicsEngine{
    public:
        ExampleEngine(HWND hWnd, UINT msPerFrame);
    protected:
        void update() override;
    private:
        SquareFactory squareFactory = SquareFactory();
        std::unique_ptr<Drawable> mySquare;
};