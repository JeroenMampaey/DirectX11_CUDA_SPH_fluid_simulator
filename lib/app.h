#pragma once

#include "window.h"

class App{
    public:
        App(const char* name, std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND));
        int Go();
        ~App() noexcept;
    private:
        Window wnd;
};