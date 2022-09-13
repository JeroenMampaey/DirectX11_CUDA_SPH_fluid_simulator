#pragma once

#include "../lib/lib_header.h"

class App : public EventListener{
    public:
        App(Window& wnd);
        void handleEvent(const Event& event) noexcept;
        void showFrame();
    
    private:
        Window& wnd;
        float dt = 0.0;
        float velocity = 0.0f;
        FilledRectangleFactory filledRectangleFactory = FilledRectangleFactory();
        LineFactory lineFactory = LineFactory();
        FilledCircleFactory filledCircleFactory = FilledCircleFactory();
        HollowRectangleFactory hollowRectangleFactory = HollowRectangleFactory();
        std::vector<std::unique_ptr<Drawable>> filledRectangles;
        std::vector<std::unique_ptr<Drawable>> lines;
        std::vector<std::unique_ptr<Drawable>> filledCircles;
        std::vector<std::unique_ptr<Drawable>> hollowRectangles;
};