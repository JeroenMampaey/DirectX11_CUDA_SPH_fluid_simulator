#ifndef UNICODE
#define UNICODE
#endif

#include <windows.h>
#include <memory>
#include "graphics.h"

#include "../extra_code/basewin.h"
#include "physics.h"
#include <wrl/client.h>
#include <memory>


class MainWindow : public BaseWindow<MainWindow>
{
private:
    std::unique_ptr<Graphics> pGfx = nullptr;

public:
    void Render();
    MainWindow(){}
    PCWSTR  ClassName() const { return L"SPH Window Class"; }
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
};

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int nCmdShow)
{
    MainWindow win;

    if (!win.Create(L"SPH", WS_CAPTION | WS_SYSMENU , 0, CW_USEDEFAULT, CW_USEDEFAULT, RIGHT-LEFT, FLOOR-CEILING))
    {
        return 0;
    }

    ShowWindow(win.Window(), nCmdShow);

    // Run the message loop.
    MSG  msg = {};

    while (WM_QUIT != msg.message)
    {
        if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
        else
        {
            // Do some rendering
            win.Render();
        }
    }

    return 0;
}

LRESULT MainWindow::HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CREATE:
        // Register this class as a window property (used for callbacks)
        SetProp(m_hwnd, L"MainWindow", this);

        pGfx = std::make_unique<Graphics>(m_hwnd);

        return 0;

    case WM_DESTROY:
        // Remove this class as a window property (used for callbacks)
        RemoveProp(m_hwnd, L"MainWindow");

        PostQuitMessage(0);

        return 0;
    }
    return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
}

void MainWindow::Render(){
    pGfx->ClearBuffer(0.1, 0.5, 1.0);
    pGfx->EndFrame();
}