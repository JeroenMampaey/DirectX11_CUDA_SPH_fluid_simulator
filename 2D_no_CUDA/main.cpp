#ifndef UNICODE
#define UNICODE
#endif

#include <windows.h>
#include <d2d1.h>
#include <thread>
#include <atomic>
#include <string>
#include <vector>

#include "../basewin.h"
#include "physics.h"

#define TIMER_ID1 0
#define TIMER_ID2 1

#define FPS_UPDATE_INTERVAL_MILI 300.0

#define DEFAULT_NUMPOINTS 1000
#define DEFAULT_NUMBOUNDARIES 3

// 10000 lines for particles: max 4+3 characters plus whitespace and '\n' -> 90000
// 500 lines for boundaries: max 4+3+4+3 characters plus 3 whitespaces and '\n' -> 9000
#define MAX_BUFFERSIZE 99000

class MainWindow : public BaseWindow<MainWindow>
{
    ID2D1Factory            *pFactory;
    ID2D1HwndRenderTarget   *pRenderTarget;
    ID2D1SolidColorBrush    *pBrush;
    std::thread            physicsThread;

    std::atomic<bool> exit = false;
    std::atomic<bool> updateRequired = false;

    std::atomic<int> drawingIndex = 0;

    int numboundaries = 0;
    int numpoints = 0;
    Boundary* boundaries;
    Particle* particles;

    int last_ms = 0;
    int passed_ms = 0;

    void buildSimulationLayout();
    void buildDefaultSimulationLayout();
    void buildSimulationLayoutFromFile(char* ReadBuffer);

    HRESULT CreateGraphicsResources();
    void DiscardGraphicsResources();
    void OnPaint();
    void Resize();

    DWORD g_BytesTransferred = 0;

public:

    MainWindow() : pFactory(NULL), pRenderTarget(NULL), pBrush(NULL){}
    PCWSTR  ClassName() const { return L"SPH Window Class"; }
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
    void setBytesTransferred(DWORD g_BytesTransferred);
    static VOID CALLBACK MainWindow::FileIOCompletionRoutine(__in  DWORD dwErrorCode, __in  DWORD dwNumberOfBytesTransfered, __in  LPOVERLAPPED lpOverlapped );
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
        for(int i = 0; i < numboundaries; i++){
            pRenderTarget->DrawLine(D2D1::Point2F(boundaries[i].x1, boundaries[i].y1), D2D1::Point2F(boundaries[i].x2, boundaries[i].y2), pBrush);
        }

        // Draw the particles
        for(int i = 0; i < numpoints; i++){
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
        SetProp(m_hwnd, L"MainWindow", this);

        buildSimulationLayout();

        if (FAILED(D2D1CreateFactory(
                D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory)))
        {
            return -1;
        }

        // Setup the background thread for all caclulations concerning the physics
        physicsThread = std::thread(physicsBackgroundThread, std::ref(exit), std::ref(updateRequired), std::ref(drawingIndex), boundaries, numboundaries, particles, numpoints, m_hwnd);

        // Setup the timer to notify the physics thread to update everything
        SetTimer(m_hwnd,
                TIMER_ID1,
                UPDATES_PER_RENDER*INTERVAL_MILI,
                (TIMERPROC) NULL);

        // Setup the timer show ms per frame in the title bar
        SetTimer(m_hwnd,
                TIMER_ID2,
                FPS_UPDATE_INTERVAL_MILI,
                (TIMERPROC) NULL);
        
        return 0;

    case WM_DESTROY:
        RemoveProp(m_hwnd, L"MainWindow");

        DiscardGraphicsResources();
        SafeRelease(&pFactory);
        PostQuitMessage(0);

        // Stop the background thread
        exit.store(true);
        drawingIndex.store(numpoints);
        physicsThread.join();

        delete[] boundaries;
        delete[] particles;

        return 0;

    case WM_PAINT:
        {
            // Estimate the number of ms per frame
            SYSTEMTIME st;
            GetSystemTime(&st);
            passed_ms = st.wMilliseconds+1000*st.wSecond-last_ms+1000*st.wMinute*60;
            last_ms = st.wMilliseconds+1000*st.wSecond+1000*st.wMinute*60;
        }

        // Repaint the screen
        OnPaint();
        return 0;

    case WM_TIMER: 
 
        switch (wParam) 
        { 
            case TIMER_ID1:
                // Notify the physics thread to request an update
                updateRequired.store(true);
                return 0;
            
            case TIMER_ID2:
                // Update the title bar with number of ms per frame
                SetWindowTextA(m_hwnd, (LPCSTR)std::to_string(passed_ms).c_str());
                return 0;
        }
    }
    return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
}

void MainWindow::buildSimulationLayout(){
    HANDLE hFile;
    DWORD  dwBytesRead = 0;
    char   ReadBuffer[MAX_BUFFERSIZE] = {0};
    OVERLAPPED ol = {0};
    ol.hEvent = (HANDLE)m_hwnd;

    hFile = CreateFileA("../simulation_layout/simulation2D.txt", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);

    if(hFile == INVALID_HANDLE_VALUE){
        buildDefaultSimulationLayout();
        return;
    }

    if( FALSE == ReadFileEx(hFile, ReadBuffer, MAX_BUFFERSIZE-1, &ol, MainWindow::FileIOCompletionRoutine) )
    {
        CloseHandle(hFile);
        buildDefaultSimulationLayout();
        return;
    }

    SleepEx(10000, TRUE);

    dwBytesRead = g_BytesTransferred;

    if (dwBytesRead > 0 && dwBytesRead <= MAX_BUFFERSIZE-1)
    {
        ReadBuffer[dwBytesRead]='\0';
        buildSimulationLayoutFromFile(ReadBuffer);
    }
    else{
        buildDefaultSimulationLayout();
    }

    CloseHandle(hFile);
}

void MainWindow::buildDefaultSimulationLayout(){
    numpoints = DEFAULT_NUMPOINTS;
    numboundaries = DEFAULT_NUMBOUNDARIES;

    particles = new Particle[numpoints];
    boundaries = new Boundary[numboundaries];

    boundaries[0] = Boundary(LEFT, FLOOR, LEFT, CEILING);
    boundaries[1] = Boundary(RIGHT, FLOOR, LEFT, FLOOR);
    boundaries[2] = Boundary(RIGHT, CEILING, RIGHT, FLOOR);

    float start_x = LEFT + 2*RADIUS;
    float end_x = LEFT + (RIGHT-LEFT)/3;
    float start_y = CEILING + 2*RADIUS;
    float end_y = CEILING + (FLOOR-CEILING)/2;
    float interval = sqrt((end_x-start_x)*(end_y-start_y)/DEFAULT_NUMPOINTS);
    float x = start_x;
    float y = start_y;
    for(int i=0; i<DEFAULT_NUMPOINTS; i++)
    {
        particles[i] = Particle(x, y, 0, 0, 0);
        y = (x+interval > end_x) ? y+interval : y;
        x = (x+interval > end_x) ? start_x : x+interval;
    }
}

void MainWindow::buildSimulationLayoutFromFile(char* ReadBuffer){
    std::vector<Particle> tempParticles;
    std::vector<Boundary> tempBoundaries;
    int index = 11;
    while(ReadBuffer[index]!='L'){
        int first_number = 0;
        while(ReadBuffer[index]!=' '){
            first_number = first_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        int second_number = 0;
        while(ReadBuffer[index]!='\n'){
            second_number = second_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        tempParticles.push_back(Particle(first_number, second_number, 0, 0, 0));
    }
    index += 7;
    while(ReadBuffer[index]!='\0'){
        int first_number = 0;
        while(ReadBuffer[index]!=' '){
            first_number = first_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        int second_number = 0;
        while(ReadBuffer[index]!=' '){
            second_number = second_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        int third_number = 0;
        while(ReadBuffer[index]!=' '){
            third_number = third_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        int fourth_number = 0;
        while(ReadBuffer[index]!='\n'){
            fourth_number = fourth_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        tempBoundaries.push_back(Boundary(first_number, second_number, third_number, fourth_number));
    }

    numpoints = tempParticles.size();
    numboundaries = tempBoundaries.size();

    particles = new Particle[numpoints];
    boundaries = new Boundary[numboundaries];

    for(int i=0; i<numpoints; i++){
        particles[i] = tempParticles[i];
    }

    for(int i=0; i<numboundaries; i++){
        boundaries[i] = tempBoundaries[i];
    }
}

VOID CALLBACK MainWindow::FileIOCompletionRoutine(__in  DWORD dwErrorCode, __in  DWORD dwNumberOfBytesTransfered, __in  LPOVERLAPPED lpOverlapped ){
    MainWindow* pThis = (MainWindow*)GetProp((HWND)lpOverlapped->hEvent, L"MainWindow");
    pThis->setBytesTransferred(dwNumberOfBytesTransfered);
}

void MainWindow::setBytesTransferred(DWORD g_BytesTransferred){
    this->g_BytesTransferred = g_BytesTransferred;
}
