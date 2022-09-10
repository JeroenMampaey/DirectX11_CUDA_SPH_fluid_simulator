#pragma once

#include "window.h"
#include "exports.h"

class LIBRARY_API App{
    public:
        App(const char* name, std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND));
        int go();
        ~App() noexcept;
    private:
        Window wnd;
};