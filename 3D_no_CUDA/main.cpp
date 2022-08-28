#ifndef UNICODE
#define UNICODE
#endif

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_3.h>
#include <memory>

#include "../extra_code/basewin.h"
#include "physics.h"
#include <wrl/client.h>


class MainWindow : public BaseWindow<MainWindow>
{
public:

    MainWindow(){}
    PCWSTR  ClassName() const { return L"SPH Window Class"; }
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
};

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int nCmdShow)
{
    MainWindow win;

    if (!win.Create(L"SPH", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, 0, CW_USEDEFAULT, CW_USEDEFAULT, RIGHT-LEFT, FLOOR-CEILING))
    {
        return 0;
    }

    ShowWindow(win.Window(), nCmdShow);

    // Run the message loop.
    bool bGotMsg;
    MSG  msg;
    msg.message = WM_NULL;
    PeekMessage(&msg, NULL, 0U, 0U, PM_NOREMOVE);

    while (WM_QUIT != msg.message)
    {
        // Process window events.
        // Use PeekMessage() so we can use idle time to render the scene. 
        bGotMsg = (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE) != 0);

        if (bGotMsg)
        {
            // Translate and dispatch the message
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            // Update the scene.
            //renderer->Update();

            // Render frames during idle time (when no messages are waiting).
            //renderer->Render();

            // Present the frame to the screen.
            //deviceResources->Present();
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

        return 0;

    case WM_DESTROY:
        // Remove this class as a window property (used for callbacks)
        RemoveProp(m_hwnd, L"MainWindow");

        PostQuitMessage(0);

        return 0;

    case WM_PAINT:
        return 0;
    }
    return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
}