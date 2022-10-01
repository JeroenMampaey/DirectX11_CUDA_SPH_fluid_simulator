#include "app.h"

#define SYNCINTERVAL 1

App::App()
    :
    wnd(createWindow("Example", SYNCINTERVAL)),
    entityManager(EntityManager()),
    eventHandlerSystem(EventHandlerSystem(wnd.getEventBus(), entityManager)),
    renderSystem(RenderSystem(wnd.getGraphicsEngine()))
{}

int App::go(){
    while(true){
        if(const std::optional<int> ecode = Window::processMessages()){
            return *ecode;
        }
        wnd.checkForThrownExceptions();
        renderSystem.update(entityManager);
    }
}