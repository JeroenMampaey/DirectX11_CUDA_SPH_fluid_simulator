#include "app.h"

#define SYNCINTERVAL 1

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow){
	try{
		Window wnd = Window("Example", SYNCINTERVAL);
		App app = App(wnd);
		while(true){
			if(const std::optional<int> ecode = Window::processMessages()){
				return *ecode;
			}
			app.showFrame();
		}
	}
	catch(const WinException& e){
		MessageBoxA(nullptr, e.what(), e.GetType(), MB_OK | MB_ICONEXCLAMATION);
	}
	catch(const std::exception& e){
		MessageBoxA(nullptr, e.what(), "Standard Exception", MB_OK | MB_ICONEXCLAMATION);
	}
	catch(...){
		MessageBoxA(nullptr, "No details available", "Unknown Exception", MB_OK | MB_ICONEXCLAMATION);
	}
	return -1;
}