#pragma once

#include "../lib/lib_header.h"

class App : public EventListener{
    public:
        App(Window& wnd);
        void periodicCallback() noexcept;
        void handleEvent(const Event& event) noexcept;
    
    private:
        void handleKeyEvent(LPARAM key) noexcept;
        float cameraPositionX = 0.0f;
        Window& wnd;
        float dt = 0.0;
        float velocity = 0.0f;
        std::vector<FilledRectangle*> filledRectangles;
        std::vector<FilledCircle*> filledCircles;
};