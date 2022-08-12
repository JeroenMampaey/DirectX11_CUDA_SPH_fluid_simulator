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
#define DEFAULT_NUMPUMPS 0

// 10000 lines for particles: max 4+3 characters plus whitespace and '\n' -> 90000
// 500 lines for boundaries: max 4+3+4+3 characters plus 3 whitespaces and '\n' -> 9000
#define MAX_BUFFERSIZE 99000

class MainWindow : public BaseWindow<MainWindow>
{
    ID2D1Factory            *pFactory;
    ID2D1HwndRenderTarget   *pRenderTarget;
    ID2D1SolidColorBrush    *pBrush;
    std::thread            physicsThread;

    std::atomic<bool> exit = {false};
    std::atomic<bool> updateRequired = {false};

    std::atomic<bool> doneDrawing = {false};

    int numboundaries = 0;
    int numpoints = 0;
    int numpumps = 0;

    Boundary* boundaries;
    Particle* particles;
    Pump* pumps;
    PumpVelocity* pumpvelocities;

    int last_ms = 0;
    int passed_ms = 0;

    bool buildSimulationLayout();
    bool buildDefaultSimulationLayout();
    bool buildSimulationLayoutFromFile(char* ReadBuffer);
    bool allocateArrayMemory();

    HRESULT CreateGraphicsResources();
    void DiscardGraphicsResources();
    void OnPaint();
    void Resize();
    void destroyApplication();

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
        }
        
        doneDrawing.store(true);

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
        // Register this class as a window property (used for callbacks)
        SetProp(m_hwnd, L"MainWindow", this);
        
        //Try to open a console for debugging
        if(DEBUG){
            Debugger::startDebugger(m_hwnd);
            Debugger::bindTopic(DEBUG_PASSED_MS_ID, "MsPerFrame");
        }

        if (FAILED(D2D1CreateFactory(
                D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory)))
        {
            destroyApplication();
            return -1;
        }

        // Setup the initial conditions for the simulation
        if(!buildSimulationLayout()){
            destroyApplication();
            return -1;
        }

        // Setup the background thread for all caclulations concerning the physics
        physicsThread = std::thread(physicsBackgroundThread, std::ref(exit), std::ref(updateRequired), std::ref(doneDrawing), boundaries, numboundaries, particles, numpoints, pumps, pumpvelocities, numpumps, m_hwnd);

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
        destroyApplication();
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
                Debugger::updateTopic(DEBUG_PASSED_MS_ID, (char*)std::to_string(passed_ms).c_str());
                return 0;
        }
    }
    return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
}

bool MainWindow::buildSimulationLayout(){
    HANDLE hFile;
    DWORD  dwBytesRead = 0;
    char   ReadBuffer[MAX_BUFFERSIZE] = {0};
    OVERLAPPED ol = {0};
    ol.hEvent = (HANDLE)m_hwnd;

    hFile = CreateFileA("../simulation_layout/simulation2D.txt", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);

    if(hFile == INVALID_HANDLE_VALUE){
        // If the file does not exist, build the default layout
        return buildDefaultSimulationLayout();
    }

    // Try reading the simulation2D.txt file
    if( FALSE == ReadFileEx(hFile, ReadBuffer, MAX_BUFFERSIZE-1, &ol, MainWindow::FileIOCompletionRoutine) )
    {
        // If something went wrong, build the default layout
        CloseHandle(hFile);
        return buildDefaultSimulationLayout();
    }

    // Wait at most 10 seconds for the file to be read (this returns early once the file has been read thus usually the 10 seconds 
    // will not be necessary)
    SleepEx(10000, TRUE);

    dwBytesRead = g_BytesTransferred;

    bool success;
    if (dwBytesRead > 0 && dwBytesRead <= MAX_BUFFERSIZE-1)
    {
        // If the file was read successfully, parse the contents
        ReadBuffer[dwBytesRead]='\0';
        success = buildSimulationLayoutFromFile(ReadBuffer);
    }
    else{
        // If something went wrong, build the default layout
        success = buildDefaultSimulationLayout();
    }

    CloseHandle(hFile);
    return success;
}

// Setup a simple default simulation layout
bool MainWindow::buildDefaultSimulationLayout(){
    numpoints = DEFAULT_NUMPOINTS;
    numboundaries = DEFAULT_NUMBOUNDARIES;
    numpumps = DEFAULT_NUMPUMPS;

    // Allocate pinned memory for the particles, boundaries and pumps
    if(!allocateArrayMemory()){
        return false;
    }

    boundaries[0].x1 = LEFT;
    boundaries[0].y1 = FLOOR;
    boundaries[0].x2 = LEFT;
    boundaries[0].y2 = CEILING;

    boundaries[1].x1 = RIGHT;
    boundaries[1].y1 = FLOOR;
    boundaries[1].x2 = LEFT;
    boundaries[1].y2 = FLOOR;
    
    boundaries[2].x1 = RIGHT;
    boundaries[2].y1 = CEILING;
    boundaries[2].x2 = RIGHT;
    boundaries[2].y2 = FLOOR;

    float start_x = LEFT + 2*RADIUS;
    float end_x = LEFT + (RIGHT-LEFT)/4;
    float start_y = CEILING + (FLOOR-CEILING)/4;
    float end_y = FLOOR - 2*RADIUS;
    float interval = sqrt((end_x-start_x)*(end_y-start_y)/DEFAULT_NUMPOINTS);
    float x = start_x;
    float y = start_y;
    for(int i=0; i<DEFAULT_NUMPOINTS; i++)
    {
        particles[i].x = x;
        particles[i].y = y;
        y = (x+interval > end_x) ? y+interval : y;
        x = (x+interval > end_x) ? start_x : x+interval;
    }
    return true;
}

// Parse the ReadBuffer and setup the corresponding simulation layout
bool MainWindow::buildSimulationLayoutFromFile(char* ReadBuffer){
    std::vector<Particle> tempParticles;
    std::vector<Boundary> tempBoundaries;
    std::vector<Pump> tempPumps;
    std::vector<PumpVelocity> tempPumpVelocities;

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
        tempParticles.push_back({(float)first_number, (float)second_number});
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
        tempBoundaries.push_back({(float)first_number, (float)second_number, (float)third_number, (float)fourth_number});
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
        tempPumps.push_back({(float)first_number, (float)second_number, (float)third_number, (float)fourth_number});
        tempPumpVelocities.push_back({(float)fifth_number, (float)sixth_number});
    }

    // Build the particle and boundary arrays from the temporary vectors
    numpoints = tempParticles.size();
    numboundaries = tempBoundaries.size();
    numpumps = tempPumps.size();

    // Allocate pinned memory for the particles, boundaries and pumps
    if(!allocateArrayMemory()){
        return false;
    }

    for(int i=0; i<numpoints; i++){
        particles[i] = tempParticles[i];
    }

    for(int i=0; i<numboundaries; i++){
        boundaries[i] = tempBoundaries[i];
    }

    for(int i=0; i<numpumps; i++){
        pumps[i] = tempPumps[i];
        pumpvelocities[i] = tempPumpVelocities[i];
    }
    return true;
}

// Callback used for reading the simulation2D.txt file
VOID CALLBACK MainWindow::FileIOCompletionRoutine(__in  DWORD dwErrorCode, __in  DWORD dwNumberOfBytesTransfered, __in  LPOVERLAPPED lpOverlapped ){
    MainWindow* pThis = (MainWindow*)GetProp((HWND)lpOverlapped->hEvent, L"MainWindow");
    // Signal the number of bytes read from the file to the MainWindow instance registered as property of the window
    pThis->setBytesTransferred(dwNumberOfBytesTransfered);
}

// Setter function
void MainWindow::setBytesTransferred(DWORD g_BytesTransferred){
    this->g_BytesTransferred = g_BytesTransferred;
}

// Destroy the application in a safe manner
void MainWindow::destroyApplication(){
    // Remove this class as a window property (used for callbacks)
    RemoveProp(m_hwnd, L"MainWindow");

    // Check if background thread is running
    if(physicsThread.joinable()){
        // Stop the background thread
        exit.store(true);
        doneDrawing.store(true);
        physicsThread.join();
    }

    // Remove graphic resources
    DiscardGraphicsResources();
    SafeRelease(&pFactory);

    // Remove all allocated pointers
    if(boundaries){
        cudaFreeHost(boundaries);
    }
    if(particles){
        cudaFreeHost(particles);
    }
    if(pumps){
        cudaFreeHost(pumps);
    }
    if(pumpvelocities){
        cudaFreeHost(pumpvelocities);
    }

    //destroy the debugging console
    if(DEBUG){
        Debugger::stopDebugger();
    }

    // Terminate the program
    PostQuitMessage(0);
}

// Allocates the memory for the particle, boundary and pump arrays
bool MainWindow::allocateArrayMemory(){
    cudaError_t status;
    if(numpoints > 0){
        status = cudaMallocHost((void**)&particles, sizeof(Particle)*numpoints);
        if (status != cudaSuccess){
            return false;
        }
    }
    if(numboundaries > 0){
        status = cudaMallocHost((void**)&boundaries, sizeof(Boundary)*numboundaries);
        if (status != cudaSuccess){
            return false;
        }
    }
    if(numpumps > 0){
        status = cudaMallocHost((void**)&pumps, sizeof(Pump)*numpumps);
        if (status != cudaSuccess){
            return false;
        }
        status = cudaMallocHost((void**)&pumpvelocities, sizeof(PumpVelocity)*numpumps);
        if (status != cudaSuccess){
            return false;
        }
    }
    return true;
}