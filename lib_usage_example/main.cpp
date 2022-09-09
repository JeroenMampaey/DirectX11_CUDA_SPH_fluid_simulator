#include "../lib/lib_header.h"
#include "example_engine.h"

#define MS_PER_FRAME 1000

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow){
	try
	{
        std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND) = [](HWND hWnd){
            std::unique_ptr<GraphicsEngine> retval = std::make_unique<ExampleEngine>(hWnd, MS_PER_FRAME);
            return retval;
        };
		return App("Example", engineFactory).Go();
	}
	catch(const WinException& e)
	{
		MessageBoxA(nullptr, e.what(), e.GetType(), MB_OK | MB_ICONEXCLAMATION);
	}
	catch(const std::exception& e)
	{
		MessageBoxA(nullptr, e.what(), "Standard Exception", MB_OK | MB_ICONEXCLAMATION);
	}
	catch(...)
	{
		MessageBoxA(nullptr, "No details available", "Unknown Exception", MB_OK | MB_ICONEXCLAMATION);
	}
	return -1;
}