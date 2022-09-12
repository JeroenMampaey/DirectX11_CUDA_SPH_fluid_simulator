#pragma once

#include "../lib/lib_header.h"

#define SYNCINTERVAL 4

class ExampleEngine : public GraphicsEngine{
    public:
        ExampleEngine(HWND hWnd, UINT syncInterval);
        void update() override;
    private:
        float dt = 0.0;
        SquareFactory squareFactory = SquareFactory();
        LineFactory lineFactory = LineFactory();
        CircleFactory circleFactory = CircleFactory();
        float square_velocity = 0.0f;
        std::vector<std::unique_ptr<Drawable>> squares;
        std::vector<std::unique_ptr<Drawable>> lines;
        std::vector<std::unique_ptr<Drawable>> circles;
};