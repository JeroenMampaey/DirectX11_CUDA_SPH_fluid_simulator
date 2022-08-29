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

#include "my_exception.h"
#include <sstream>

class MainWindow : public BaseWindow<MainWindow>
{
    private:
        std::unique_ptr<Graphics> pGfx = nullptr;

    public:
        class Exception : public MyException{
            public:
                Exception(int line, const char* file, HRESULT hr) noexcept;
                const char* what() const noexcept override;
                const char* GetType() const noexcept override;
                static std::string TranslateErrorCode(HRESULT hr) noexcept;
                HRESULT GetErrorCode() const noexcept;
                std::string GetErrorString() const noexcept;
            private:
                HRESULT hr;
        };

        void Render();
        MainWindow(){};
        PCWSTR  ClassName() const { return L"SPH Window Class"; };
        LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
};

#define CHWND_EXCEPT(hr) MainWindow::Exception(__LINE__, __FILE__, hr)

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

MainWindow::Exception::Exception(int line, const char* file, HRESULT hr) noexcept
    :
    MyException(line, file),
    hr(hr)
{}

const char* MainWindow::Exception::what() const noexcept{
    std::ostringstream oss;
    oss << GetType() << std::endl
        << "[Error Code]" << GetErrorCode() << std::endl
        << "[Description]" << GetErrorString() << std::endl
        << GetOriginString();
    whatBuffer = oss.str();
    return whatBuffer.c_str();
}

const char* MainWindow::Exception::GetType() const noexcept{
    return "MainWindow::Exception";
}

std::string MainWindow::Exception::TranslateErrorCode(HRESULT hr) noexcept{
    char* pMsgBuf = nullptr;
    DWORD nMsgLen = FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        hr,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPWSTR>(&pMsgBuf),
        0,
        nullptr
    );
    if(nMsgLen==0){
        return "Unidentified error code";
    }
    std::string errorString = pMsgBuf;
    LocalFree(pMsgBuf);
    return errorString;
}

HRESULT MainWindow::Exception::GetErrorCode() const noexcept{
    return hr;
}

std::string MainWindow::Exception::GetErrorString() const noexcept{
    return TranslateErrorCode(hr);
}