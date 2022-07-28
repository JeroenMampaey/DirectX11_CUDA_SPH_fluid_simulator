#ifndef UNICODE
#define UNICODE
#endif

#include <windows.h>
#include <d2d1.h>
#include <thread>
#include <atomic>

#include "basewin.h"
#include "physics.h"

#define TIMER_ID 0

// safely release the COM interface pointers as recommended here:
// https://docs.microsoft.com/en-us/windows/win32/medfound/saferelease
template <class T> void SafeRelease(T **ppT)
{
    if (*ppT)
    {
        (*ppT)->Release();
        *ppT = NULL;
    }
}

class MainWindow : public BaseWindow<MainWindow>
{
    ID2D1Factory            *pFactory;
    ID2D1HwndRenderTarget   *pRenderTarget;
    ID2D1SolidColorBrush    *pBrush;
    D2D1_ELLIPSE            ellipse;
    std::thread            physicsThread;
    int height = 0;

    std::atomic<bool> exit = false;
    std::atomic<bool> updateRequired = false;

    std::atomic<int> drawingIndex = 0;

    Boundary boundaries[NUMBOUNDARIES];
    Particle particles[NUMPOINTS];

    HRESULT CreateGraphicsResources();
    void    DiscardGraphicsResources();
    void    OnPaint();
    void    Resize();

public:

    MainWindow() : pFactory(NULL), pRenderTarget(NULL), pBrush(NULL)
    {
        // Initialize the boundaries
        // Point1 and point2 must be specified in the specific order so that the normal is pointing to the outside (a.k.a NOT towards 
        // the particles)
        boundaries[0] = Boundary(LEFT, FLOOR, LEFT, CEILING);
        boundaries[1] = Boundary(RIGHT, FLOOR, LEFT, FLOOR);
        boundaries[2] = Boundary(RIGHT, CEILING, RIGHT, FLOOR);

        // Initialize the particles
        float start_x = LEFT + 2*RADIUS;
        float end_x = RIGHT - 2*RADIUS;
        float start_y = CEILING + 2*RADIUS;
        float end_y = (FLOOR-CEILING)/2;
        float interval = sqrt((end_x-start_x)*(end_y-start_y)/NUMPOINTS);
        float x = start_x;
        float y = start_y;
        for(int i=0; i<NUMPOINTS; i++)
        {
            particles[i] = Particle(x, y, 0, 0, 0);
            y = (x+interval > end_x) ? y+interval : y;
            x = (x+interval > end_x) ? start_x : x+interval;
        }
    }

    PCWSTR  ClassName() const { return L"Circle Window Class"; }
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
};

HRESULT MainWindow::CreateGraphicsResources()
{
    // Make sure that the graphics resources have been created
    HRESULT hr = S_OK;
    if (pRenderTarget == NULL)
    {
        RECT rc;
        GetClientRect(m_hwnd, &rc);

        D2D1_SIZE_U size = D2D1::SizeU(rc.right, rc.bottom);

        hr = pFactory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(),
            D2D1::HwndRenderTargetProperties(m_hwnd, size),
            &pRenderTarget);

        if (SUCCEEDED(hr))
        {
            const D2D1_COLOR_F color = D2D1::ColorF(1.0f, 1.0f, 0);
            hr = pRenderTarget->CreateSolidColorBrush(color, &pBrush);
        }
    }
    return hr;
}

void MainWindow::DiscardGraphicsResources()
{
    SafeRelease(&pRenderTarget);
    SafeRelease(&pBrush);
}

void MainWindow::OnPaint()
{
    // first make sure graphics resources are created already
    HRESULT hr = CreateGraphicsResources();
    if (SUCCEEDED(hr))
    {
        PAINTSTRUCT ps;
        BeginPaint(m_hwnd, &ps);
     
        pRenderTarget->BeginDraw();

        pRenderTarget->Clear( D2D1::ColorF(D2D1::ColorF::SkyBlue) );
        
        // Draw the boundaries
        for(int i = 0; i < NUMBOUNDARIES; i++){
            pRenderTarget->DrawLine(D2D1::Point2F(boundaries[i].x1, boundaries[i].y1), D2D1::Point2F(boundaries[i].x2, boundaries[i].y2), pBrush);
        }

        // Draw the particles
        for(int i = 0; i < NUMPOINTS; i++){
            pRenderTarget->FillEllipse(D2D1::Ellipse(D2D1::Point2F(particles[i].x, particles[i].y), RADIUS, RADIUS), pBrush);
            drawingIndex.store(i+1);
        }

        hr = pRenderTarget->EndDraw();
        if (FAILED(hr) || hr == D2DERR_RECREATE_TARGET)
        {
            DiscardGraphicsResources();
        }
        EndPaint(m_hwnd, &ps);
    }
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int nCmdShow)
{
    MainWindow win;

    if (!win.Create(L"Circle", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, 0, CW_USEDEFAULT, CW_USEDEFAULT, RIGHT-LEFT, FLOOR-CEILING))
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
        if (FAILED(D2D1CreateFactory(
                D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory)))
        {
            return -1;
        }

        // Setup the background thread for all caclulations concerning the physics
        physicsThread = std::thread(physicsBackgroundThread, std::ref(exit), std::ref(updateRequired), std::ref(drawingIndex), boundaries, particles, m_hwnd);

        // Setup the timer to notify the physics thread to update everything
        SetTimer(m_hwnd,
                TIMER_ID,
                INTERVAL_MILI,
                (TIMERPROC) NULL);
        return 0;

    case WM_DESTROY:
        DiscardGraphicsResources();
        SafeRelease(&pFactory);
        PostQuitMessage(0);

        // Stop the background thread
        exit.store(true);
        physicsThread.join();

        return 0;

    case WM_PAINT:
        // Repaint the screen
        OnPaint();
        return 0;

    case WM_TIMER: 
 
        switch (wParam) 
        { 
            case TIMER_ID:
                // Notify the physics thread to request an update
                updateRequired.store(true);
                return 0;
        }
    }
    return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
}