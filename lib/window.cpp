#include "window.h"


Window::WindowClass Window::WindowClass::wndClass;

Window::WindowClass::WindowClass() noexcept : hInst(GetModuleHandleW(nullptr)){
	WNDCLASSEXA wc = {0};
	wc.cbSize = sizeof(wc);
	wc.style = CS_OWNDC;
	wc.lpfnWndProc = HandleMsgSetup;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = GetInstance();
	wc.hCursor = nullptr;
	wc.hbrBackground = nullptr;
	wc.lpszMenuName = nullptr;
	wc.lpszClassName = GetName();

	RegisterClassExA(&wc);
}

Window::WindowClass::~WindowClass() noexcept{
	UnregisterClassA(wndClassName, GetInstance());
}

const char* Window::WindowClass::GetName() noexcept{
	return wndClassName;
}

HINSTANCE Window::WindowClass::GetInstance() noexcept{
	return wndClass.hInst;
}

// Window Stuff
Window::Window(const char* name, std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND)){
	RECT wr;
	wr.left = 100;
	wr.right = WIDTH + wr.left;
	wr.top = 100;
	wr.bottom = HEIGHT + wr.top;
	if(AdjustWindowRect(&wr, WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, FALSE)==0){
		throw CHWND_LAST_EXCEPT();
	}
	
	hWnd = CreateWindowExA(
		0L, WindowClass::GetName(),name,
		WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU,
		CW_USEDEFAULT,CW_USEDEFAULT,wr.right - wr.left,wr.bottom - wr.top,
		nullptr,nullptr,WindowClass::GetInstance(),this
	);
	
	if( hWnd == nullptr )
	{
		throw CHWND_LAST_EXCEPT();
	}

	ShowWindow(hWnd, SW_SHOWDEFAULT);

    pGfx = engineFactory(hWnd);

    SetWindowTextA(hWnd, name);
}

Window::~Window() noexcept{
	DestroyWindow(hWnd);
}

std::optional<int> Window::ProcessMessages() noexcept{
	MSG msg;

	while(PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)){
		if(msg.message==WM_QUIT){
			return (int)msg.wParam;
		}

		TranslateMessage(&msg);
		DispatchMessageW(&msg);
	}

	return {};
}

LRESULT CALLBACK Window::HandleMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept{
	if(msg==WM_NCCREATE){
		const CREATESTRUCTW* const pCreate = reinterpret_cast<CREATESTRUCTW*>(lParam);
		Window* const pWnd = static_cast<Window*>(pCreate->lpCreateParams);
		SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWnd));
		SetWindowLongPtrW(hWnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&Window::HandleMsgThunk));
		return pWnd->HandleMsg(hWnd, msg, wParam, lParam);
	}
	return DefWindowProcW( hWnd,msg,wParam,lParam );
}

LRESULT CALLBACK Window::HandleMsgThunk(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept{
	Window* const pWnd = reinterpret_cast<Window*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
	return pWnd->HandleMsg(hWnd, msg, wParam, lParam);
}

LRESULT Window::HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept{
	switch(msg)
	{
		case WM_CLOSE:
			PostQuitMessage(0);
			return 0;
	}

	return DefWindowProcW(hWnd, msg, wParam, lParam);
}