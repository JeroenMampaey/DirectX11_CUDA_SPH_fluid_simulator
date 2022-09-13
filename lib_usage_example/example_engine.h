#pragma once

#include "../lib/lib_header.h"

#define SYNCINTERVAL 1

class ExampleEngine : public GraphicsEngine, public EventListener{
    public:
        ExampleEngine(HWND hWnd, std::shared_ptr<EventBus> pEventBus);
        void update() override;
        void handleEvent(const Event& event) noexcept override;
    private:
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