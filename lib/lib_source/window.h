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
                static const char* GetName() noexcept;
                static HINSTANCE GetInstance() noexcept;
            private:
                WindowClass() noexcept;
                ~WindowClass() noexcept;
                static constexpr const char* wndClassName = "MyWindow";
                static WindowClass wndClass;
                HINSTANCE hInst;
        };
        static LRESULT CALLBACK HandleMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;
	    static LRESULT CALLBACK HandleMsgThunk(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;
	    LRESULT HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;
        HWND hWnd;
        std::unique_ptr<GraphicsEngine> pGfx;

    public:
        Window(const char* name, std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND));
	    ~Window() noexcept;
        static std::optional<int> ProcessMessages() noexcept;
        void checkForExceptions() const;
};