#pragma once

#include <optional>
#include <memory>
#include "graphics_engine.h"
#include "event.h"
#include "events/events_includes.h"

class LIBRARY_API Window
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
        std::exception_ptr thrownException = nullptr;

    public:
        Window(const char* name, UINT syncInterval);
	    ~Window() noexcept;
        static std::optional<int> processMessages() noexcept;
        std::shared_ptr<EventBus> getEventBus() const noexcept;
        GraphicsEngine& getGraphicsEngine() const noexcept;
        void checkForThrownExceptions() const;
};