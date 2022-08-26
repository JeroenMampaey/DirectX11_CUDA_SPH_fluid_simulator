#ifndef UNICODE
#define UNICODE
#endif

#include <windows.h>
#include <d2d1.h>
#include <thread>
#include <atomic>
#include <string>
#include <vector>

#include "../extra_code/basewin.h"
#include "physics.h"

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
    MSG msg = { };
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
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