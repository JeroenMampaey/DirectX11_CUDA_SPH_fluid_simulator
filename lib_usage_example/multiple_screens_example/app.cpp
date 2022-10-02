#include "app.h"

#define SYNCINTERVAL 1

App::App()
    :
    wnd1(createWindow("Example1", SYNCINTERVAL)),
    wnd2(createWindow("Example2", SYNCINTERVAL)),
    entityManager(EntityManager()),
    eventHandlerSystem(EventHandlerSystem(wnd1.getEventBus(), wnd2.getEventBus())),
    renderSystem(RenderSystem(wnd1.getGraphicsEngine(), wnd2.getGraphicsEngine()))
{}

int App::go(){
    while(true){
        if(const std::optional<int> ecode = Window::processMessages()){
            return *ecode;
        }
        wnd1.checkForThrownExceptions();
        wnd2.checkForThrownExceptions();
        updateSystems();
    }
}

void App::updateSystems(){
    eventHandlerSystem.update(entityManager);
    renderSystem.update(entityManager);
}