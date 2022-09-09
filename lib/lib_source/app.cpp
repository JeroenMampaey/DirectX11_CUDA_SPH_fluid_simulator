#include "app.h"

App::App(const char* name, std::unique_ptr<GraphicsEngine> (*engineFactory)(HWND)) : wnd(name, engineFactory)
{}

App::~App() noexcept
{}

int App::Go(){
	while( true )
	{
		if( const auto ecode = Window::ProcessMessages() )
		{
			return *ecode;
		}
	}
}