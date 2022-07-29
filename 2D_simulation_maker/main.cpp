#ifndef UNICODE
#define UNICODE
#endif

#include <windows.h>
#include <d2d1.h>

#include "../basewin.h"
#include "simulation_builder.h"

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

    HRESULT CreateGraphicsResources();
    void    DiscardGraphicsResources();
    void    OnPaint();
    void    Resize();

    SimulationBuilder simulation_builder;


public:

    MainWindow() : pFactory(NULL), pRenderTarget(NULL), pBrush(NULL), simulation_builder(NULL)
    {
    }

    PCWSTR  ClassName() const { return L"SPH Window Class"; }
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

        pBrush->SetColor(D2D1::ColorF(D2D1::ColorF::Blue));

        std::vector<D2D1_ELLIPSE> &particles = simulation_builder.getParticles();
        for (D2D1_ELLIPSE particle : particles)
        {
            pRenderTarget->FillEllipse(particle, pBrush);
        }

        std::vector<std::pair<D2D1_POINT_2F, D2D1_POINT_2F>> &lines = simulation_builder.getLines();
        for (std::pair<D2D1_POINT_2F, D2D1_POINT_2F> line : lines)
        {
            pBrush->SetColor(D2D1::ColorF(D2D1::ColorF::Black));
            pRenderTarget->DrawLine(line.first, line.second, pBrush);
            float length = sqrt((line.second.x-line.first.x)*(line.second.x-line.first.x)+(line.second.y-line.first.y)*(line.second.y-line.first.y));
            if(length > 0){
                float nx = (line.second.y-line.first.y)/length;
                float ny = (line.first.x-line.second.x)/length;
                D2D1_POINT_2F first_point = D2D1::Point2F(line.first.x+(line.second.x-line.first.x)/2, line.first.y+(line.second.y-line.first.y)/2);
                D2D1_POINT_2F second_point = D2D1::Point2F(first_point.x+nx*30, first_point.y+ny*30);
                pBrush->SetColor(D2D1::ColorF(D2D1::ColorF::Red));
                pRenderTarget->DrawLine(first_point, second_point, pBrush);
            }
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
        if (FAILED(D2D1CreateFactory(
                D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory)))
        {
            return -1;
        }
        
        simulation_builder = SimulationBuilder(m_hwnd);

        return 0;

    case WM_DESTROY:
        DiscardGraphicsResources();
        SafeRelease(&pFactory);
        PostQuitMessage(0);

        return 0;

    case WM_PAINT:
        // Repaint the screen
        OnPaint();
        return 0;
    
    case WM_LBUTTONDOWN:
        simulation_builder.mouseEvent(LOWORD(lParam), HIWORD(lParam), MOUSE_LEFT);
        return 0;

    case WM_RBUTTONDOWN:
        simulation_builder.mouseEvent(LOWORD(lParam), HIWORD(lParam), MOUSE_RIGHT);
        return 0;
    
    case WM_MOUSEMOVE:
        simulation_builder.mouseEvent(LOWORD(lParam), HIWORD(lParam), MOUSE_MOV);
        return 0;
    
    case WM_KEYDOWN:
        simulation_builder.keyboardEvent(wParam);
    }
    return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
}