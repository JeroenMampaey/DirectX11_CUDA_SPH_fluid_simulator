#include "../lib/lib_header.h"
#include "example_engine.h"

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow){
	try
	{
        std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND) = [](HWND hWnd){
            std::unique_ptr<GraphicsEngine> retval = std::make_unique<ExampleEngine>(hWnd, SYNCINTERVAL);
            return retval;
        };
		return App("Example", engineFactory).go();
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