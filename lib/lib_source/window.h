#pragma once

#include <optional>
#include <memory>
#include "graphics_engine.h"
#include "events/events_includes.h"

class Window
{
        friend LIBRARY_API Window createWindow(const char* name, UINT syncInterval);
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

        Window(const char* name, UINT syncInterval);
        
        static LRESULT CALLBACK handleMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;
	    static LRESULT CALLBACK handleMsgThunk(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;

	    LRESULT handleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;
        HWND hWnd;
        std::shared_ptr<EventBus> pEventBus;
        std::unique_ptr<GraphicsEngine> pGfx;
        std::exception_ptr thrownException = nullptr;

    public:
	    LIBRARY_API ~Window() noexcept;
        LIBRARY_API static std::optional<int> processMessages() noexcept;
        LIBRARY_API std::shared_ptr<EventBus> getEventBus() const noexcept;
        LIBRARY_API GraphicsEngine& getGraphicsEngine() noexcept;
        LIBRARY_API void checkForThrownExceptions() const;
};