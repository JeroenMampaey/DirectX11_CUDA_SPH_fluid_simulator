#include "app.h"

App::App(const char* name, std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND,std::shared_ptr<EventBus>)) 
	: 
	wnd(name, engineFactory)
{}

App::~App() noexcept
{}

int App::go(){
	while(true){
		if(const auto ecode = Window::processMessages()){
			return *ecode;
		}
		wnd.updateGraphics();
	}
}