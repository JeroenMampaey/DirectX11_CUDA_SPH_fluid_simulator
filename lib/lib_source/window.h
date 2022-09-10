#pragma once

#include <windows.h>
#include <optional>
#include "graphics_engine.h"
#include <memory>
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib, "ole32.lib")

class Window
{
    private:
        class WindowClass
        {
            public:
                static const char* getName() noexcept;
                static HINSTANCE getInstance() noexcept;
            private:
                WindowClass() noexcept;
                ~WindowClass() noexcept;
                static constexpr const char* wndClassName = "MyWindow";
                static WindowClass wndClass;
                HINSTANCE hInst;
        };
        static LRESULT CALLBACK handleMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;
	    static LRESULT CALLBACK handleMsgThunk(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;
	    LRESULT handleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;
        HWND hWnd;
        std::unique_ptr<GraphicsEngine> pGfx;

    public:
        Window(const char* name, std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND));
	    ~Window() noexcept;
        static std::optional<int> processMessages() noexcept;
        void checkForExceptions() const;
};