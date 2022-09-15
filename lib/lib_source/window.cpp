#include "window.h"


Window::WindowClass Window::WindowClass::wndClass;

Window::WindowClass::WindowClass() noexcept : hInst(GetModuleHandleW(nullptr)){
	WNDCLASSEXA wc = {0};
	wc.cbSize = sizeof(wc);
	wc.style = CS_OWNDC;
	wc.lpfnWndProc = handleMsgSetup;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = getInstance();
	wc.hCursor = nullptr;
	wc.hbrBackground = nullptr;
	wc.lpszMenuName = nullptr;
	wc.lpszClassName = getName();

	RegisterClassExA(&wc);
}

Window::WindowClass::~WindowClass() noexcept{
	UnregisterClassA(wndClassName, getInstance());
}

const char* Window::WindowClass::getName() noexcept{
	return wndClassName;
}

HINSTANCE Window::WindowClass::getInstance() noexcept{
	return wndClass.hInst;
}

// Window Stuff
Window::Window(const char* name, UINT syncInterval){
	RECT wr;
	wr.left = 100;
	wr.right = WIDTH + wr.left;
	wr.top = 100;
	wr.bottom = HEIGHT + wr.top;
	if(AdjustWindowRect(&wr, WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, FALSE)==0){
		throw CHWND_LAST_EXCEPT();
	}
	
	hWnd = CreateWindowExA(
		0L, WindowClass::getName(),name,
		WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU,
		CW_USEDEFAULT,CW_USEDEFAULT,wr.right - wr.left,wr.bottom - wr.top,
		nullptr,nullptr,WindowClass::getInstance(),this
	);
	
	if( hWnd == nullptr )
	{
		throw CHWND_LAST_EXCEPT();
	}

	ShowWindow(hWnd, SW_SHOWDEFAULT);

	SetWindowTextA(hWnd, "Binding graphics to the window...");

	pEventBus = std::make_shared<EventBus>();

    pGfx = std::make_unique<GraphicsEngine>(hWnd, syncInterval);

	SetWindowTextA(hWnd, "Window setup was succesfull");
}

Window::~Window() noexcept{
	DestroyWindow(hWnd);
}

std::optional<int> Window::processMessages() noexcept{
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

LRESULT CALLBACK Window::handleMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept{
	if(msg==WM_NCCREATE){
		const CREATESTRUCTW* const pCreate = reinterpret_cast<CREATESTRUCTW*>(lParam);
		Window* const pWnd = static_cast<Window*>(pCreate->lpCreateParams);
		SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWnd));
		SetWindowLongPtrW(hWnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&Window::handleMsgThunk));
		return pWnd->handleMsg(hWnd, msg, wParam, lParam);
	}
	return DefWindowProcW( hWnd,msg,wParam,lParam );
}

LRESULT CALLBACK Window::handleMsgThunk(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept{
	Window* const pWnd = reinterpret_cast<Window*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
	return pWnd->handleMsg(hWnd, msg, wParam, lParam);
}

LRESULT Window::handleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept{
	try{
		switch(msg)
		{
			case WM_CLOSE:
				PostQuitMessage(0);
				return 0;
			case WM_MOUSEMOVE:
				pEventBus->post(MouseMoveEvent{LOWORD(lParam), HIWORD(lParam)});
				break;
			case WM_LBUTTONDOWN:
				pEventBus->post(MouseLeftClickEvent{LOWORD(lParam), HIWORD(lParam)});
				break;
			case WM_RBUTTONDOWN:
				pEventBus->post(MouseRightClickEvent{LOWORD(lParam), HIWORD(lParam)});
				break;
			case WM_KEYDOWN:
				pEventBus->post(KeyboardKeydownEvent{wParam});
				break;
		}

		return DefWindowProcW(hWnd, msg, wParam, lParam);
	}
	catch(...){
		thrownException = std::current_exception();
		return 0;
	}
}

std::shared_ptr<EventBus> Window::getEventBus() const noexcept{
	return pEventBus;
}

GraphicsEngine& Window::getGraphicsEngine() const noexcept{
	return *pGfx;
}

void Window::checkForThrownExceptions() const{
	if(thrownException){
		std::rethrow_exception(thrownException);
	}
}