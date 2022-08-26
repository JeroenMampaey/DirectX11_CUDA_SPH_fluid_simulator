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

#define TIMER_ID1 0
#define TIMER_ID2 1

#define FPS_UPDATE_INTERVAL_MILI 300.0

#define DEFAULT_NUMPOINTS 1500
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
    int numpumps = 0;

    Boundary* boundaries;
    Particle* particles;
    Pump* pumps;

    int last_ms = 0;
    int passed_ms = 0;

    void buildSimulationLayout();
    void buildDefaultSimulationLayout();
    void buildSimulationLayoutFromFile(char* ReadBuffer);

    HRESULT CreateGraphicsResources();
    void DiscardGraphicsResources();
    void OnPaint();

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
        pBrush->SetColor(D2D1::ColorF(D2D1::ColorF::Black));
        for(int i = 0; i < numboundaries; i++){
            pRenderTarget->DrawLine(D2D1::Point2F(boundaries[i].x1, boundaries[i].y1), D2D1::Point2F(boundaries[i].x2, boundaries[i].y2), pBrush);
        }

        // Draw the particles
        pBrush->SetColor(D2D1::ColorF(D2D1::ColorF::Blue));
        for(int i = 0; i < numpoints; i++){
            pRenderTarget->FillEllipse(D2D1::Ellipse(D2D1::Point2F(particles[i].x, particles[i].y), RADIUS, RADIUS), pBrush);
            drawingIndex.store(i+1);
        }

        // Draw the pumps
        pBrush->SetColor(D2D1::ColorF(D2D1::ColorF::Green));
        for(int i=0; i < numpumps; i++){
            pRenderTarget->DrawRectangle(D2D1::RectF(pumps[i].x_low, pumps[i].y_low, pumps[i].x_high, pumps[i].y_high), pBrush);
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
        //Try to open a console for debugging
        if(DEBUG){
            Debugger::startDebugger(m_hwnd);
            Debugger::bindTopic(DEBUG_PASSED_MS_ID, "MsPerFrame");
            Debugger::bindTopic(DEBUG_MAX_NEIGHBOURS_ID, "MaxBoundaryParticles");
        }

        if (FAILED(D2D1CreateFactory(
                D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory)))
        {
            return -1;
        }

        // Register this class as a window property (used for callbacks)
        SetProp(m_hwnd, L"MainWindow", this);

        // Setup the initial conditions for the simulation
        buildSimulationLayout();

        // Setup the background thread for all caclulations concerning the physics
        physicsThread = std::thread(physicsBackgroundThread, std::ref(exit), std::ref(updateRequired), std::ref(drawingIndex), boundaries, numboundaries, particles, numpoints, pumps, numpumps, m_hwnd);

        // Setup the timer to notify the physics thread to update everything
        SetTimer(m_hwnd,
                TIMER_ID1,
                UPDATES_PER_RENDER*INTERVAL_MILI,
                (TIMERPROC) NULL);

        // Setup the timer show ms per frame in case debugging is enabled
        if(DEBUG){
            SetTimer(m_hwnd,
                TIMER_ID2,
                FPS_UPDATE_INTERVAL_MILI,
                (TIMERPROC) NULL);
        }
        
        return 0;

    case WM_DESTROY:
        // Remove this class as a window property (used for callbacks)
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
        delete[] pumps;

        //destroy the console
        if(DEBUG){
            Debugger::stopDebugger();
        }

        return 0;

    case WM_PAINT:
        if(DEBUG){
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
                // Print the number of ms per frame to the console
                Debugger::updateTopic(DEBUG_PASSED_MS_ID, std::to_string(passed_ms).c_str());
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
        // If the file does not exist, build the default layout
        buildDefaultSimulationLayout();
        return;
    }

    // Try reading the simulation2D.txt file
    if( FALSE == ReadFileEx(hFile, ReadBuffer, MAX_BUFFERSIZE-1, &ol, MainWindow::FileIOCompletionRoutine) )
    {
        // If something went wrong, build the default layout
        CloseHandle(hFile);
        buildDefaultSimulationLayout();
        return;
    }

    // Wait at most 10 seconds for the file to be read (this returns early once the file has been read thus usually the 10 seconds 
    // will not be necessary)
    SleepEx(10000, TRUE);

    dwBytesRead = g_BytesTransferred;

    if (dwBytesRead > 0 && dwBytesRead <= MAX_BUFFERSIZE-1)
    {
        // If the file was read successfully, parse the contents
        ReadBuffer[dwBytesRead]='\0';
        buildSimulationLayoutFromFile(ReadBuffer);
    }
    else{
        // If something went wrong, build the default layout
        buildDefaultSimulationLayout();
    }

    CloseHandle(hFile);
}

// Setup a simple default simulation layout
void MainWindow::buildDefaultSimulationLayout(){
    numpoints = DEFAULT_NUMPOINTS;
    numboundaries = DEFAULT_NUMBOUNDARIES;

    particles = new Particle[numpoints];
    boundaries = new Boundary[numboundaries];

    boundaries[0] = Boundary(LEFT, FLOOR, LEFT, CEILING);
    boundaries[1] = Boundary(RIGHT, FLOOR, LEFT, FLOOR);
    boundaries[2] = Boundary(RIGHT, CEILING, RIGHT, FLOOR);

    float start_x = LEFT + 2*RADIUS;
    float end_x = LEFT + (RIGHT-LEFT)/4;
    float start_y = CEILING + (FLOOR-CEILING)/4;
    float end_y = FLOOR - 2*RADIUS;
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

// Parse the ReadBuffer and setup the corresponding simulation layout
void MainWindow::buildSimulationLayoutFromFile(char* ReadBuffer){
    std::vector<Particle> tempParticles;
    std::vector<Boundary> tempBoundaries;
    std::vector<Pump> tempPumps;

    // Skip the first line which should contain "PARTICLES:\n"
    int index = 10;
    if(ReadBuffer[index]=='\r') index++;
    index++;
    
    // Parse all particles until "LINES:\n" is found
    while(ReadBuffer[index]!='L'){
        // Read the x coordinate of the particle
        int first_number = 0;
        while(ReadBuffer[index]!=' '){
            first_number = first_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        // Read the y coordinate of the particle
        int second_number = 0;
        while(ReadBuffer[index]!='\n' && ReadBuffer[index]!='\r'){
            second_number = second_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        if(ReadBuffer[index]=='\r') index++;
        index++;
        tempParticles.push_back(Particle(first_number, second_number, 0, 0, 0));
    }

    // Skip "LINES:\n"
    index += 6;
    if(ReadBuffer[index]=='\r') index++;
    index++;

    // Parse all boundaries until "PUMPS:\n" is found
    while(ReadBuffer[index]!='P'){
        // Read the x coordinate of the first point of the line
        int first_number = 0;
        while(ReadBuffer[index]!=' '){
            first_number = first_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        // Read the y coordinate of the first point of the line
        int second_number = 0;
        while(ReadBuffer[index]!=' '){
            second_number = second_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        // Read the x coordinate of the second point of the line
        int third_number = 0;
        while(ReadBuffer[index]!=' '){
            third_number = third_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        // Read the y coordinate of the second point of the line
        int fourth_number = 0;
        while(ReadBuffer[index]!='\n' && ReadBuffer[index]!='\r'){
            fourth_number = fourth_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        if(ReadBuffer[index]=='\r') index++;
        index++;
        tempBoundaries.push_back(Boundary(first_number, second_number, third_number, fourth_number));
    }

    // Skip "PUMPS:\n"
    index += 6;
    if(ReadBuffer[index]=='\r') index++;
    index++;

    // Parse all boundaries until the end of the array (marked by '\0')
    while(ReadBuffer[index]!='\0'){
        // Read the lowest x coordinate of the pump rectangle
        int first_number = 0;
        while(ReadBuffer[index]!=' '){
            first_number = first_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        // Read the highest x coordinate of the pump rectangle
        int second_number = 0;
        while(ReadBuffer[index]!=' '){
            second_number = second_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        // Read the lowest y coordinate of the pump rectangle
        int third_number = 0;
        while(ReadBuffer[index]!=' '){
            third_number = third_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        // Read the highest y coordinate of the pump rectangle
        int fourth_number = 0;
        while(ReadBuffer[index]!=' '){
            fourth_number = fourth_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        index++;
        // Read the x coordinate of the velocity vector of the pump
        int fifth_number = 0;
        int sign = 1;
        if(ReadBuffer[index]=='-'){
            sign = -1;
            index++;
        }
        while(ReadBuffer[index]!=' '){
            fifth_number = fifth_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        fifth_number *= sign;
        index++;
        // Read the y coordinate of the velocity vector of the pump
        int sixth_number = 0;
        sign = 1;
        if(ReadBuffer[index]=='-'){
            sign = -1;
            index++;
        }
        while(ReadBuffer[index]!='\n' && ReadBuffer[index]!='\r'){
            sixth_number = sixth_number*10 + (ReadBuffer[index]-'0');
            index++;
        }
        sixth_number *= sign;
        if(ReadBuffer[index]=='\r') index++;
        index++;
        tempPumps.push_back(Pump(first_number, second_number, third_number, fourth_number, fifth_number, sixth_number));
    }

    // Build the particle and boundary arrays from the temporary vectors
    numpoints = tempParticles.size();
    numboundaries = tempBoundaries.size();
    numpumps = tempPumps.size();

    particles = new Particle[numpoints];
    boundaries = new Boundary[numboundaries];
    pumps = new Pump[numpumps];

    for(int i=0; i<numpoints; i++){
        particles[i] = tempParticles[i];
    }

    for(int i=0; i<numboundaries; i++){
        boundaries[i] = tempBoundaries[i];
    }

    for(int i=0; i<numpumps; i++){
        pumps[i] = tempPumps[i];
    }
}

// Callback used for reading the simulation2D.txt file
VOID CALLBACK MainWindow::FileIOCompletionRoutine(__in  DWORD dwErrorCode, __in  DWORD dwNumberOfBytesTransfered, __in  LPOVERLAPPED lpOverlapped ){
    MainWindow* pThis = (MainWindow*)GetProp((HWND)lpOverlapped->hEvent, L"MainWindow");
    // Signal the number of bytes read from the file to the MainWindow instance registered as property of the window
    pThis->setBytesTransferred(dwNumberOfBytesTransfered);
}

void MainWindow::setBytesTransferred(DWORD g_BytesTransferred){
    this->g_BytesTransferred = g_BytesTransferred;
}
