#pragma once

#include <windows.h>
#include <optional>
#include <memory>
#include "graphics_engine.h"
#include "event.h"
#include "events/events_includes.h"
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
        std::shared_ptr<EventBus> pEventBus;
        std::unique_ptr<GraphicsEngine> pGfx;

    public:
        Window(const char* name, std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND,std::shared_ptr<EventBus>));
	    ~Window() noexcept;
        static std::optional<int> processMessages() noexcept;
        void updateGraphics();
};