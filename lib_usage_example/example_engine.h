#pragma once

#define MS_PER_FRAME 30

#include "../lib/lib_header.h"

class ExampleEngine : public GraphicsEngine{
    public:
        ExampleEngine(HWND hWnd, UINT msPerFrame);
    protected:
        void update() override;
    private:
        SquareFactory squareFactory = SquareFactory();
        float square_velocity = 0.0f;
        std::vector<std::unique_ptr<Drawable>> squares;
};