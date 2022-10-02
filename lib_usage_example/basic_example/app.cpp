#include "app.h"

#define SYNCINTERVAL 1

App::App()
    :
    wnd(createWindow("Example", SYNCINTERVAL)),
    entityManager(EntityManager()),
    physicsSystem(PhysicsSystem(wnd.getGraphicsEngine())),
    renderSystem(RenderSystem(wnd.getGraphicsEngine())),
    eventHandlerSystem(EventHandlerSystem(wnd.getEventBus()))
{}

int App::go(){
    while(true){
        if(const std::optional<int> ecode = Window::processMessages()){
            return *ecode;
        }
        wnd.checkForThrownExceptions();
        updateSystems();
    }
}

void App::updateSystems(){
    eventHandlerSystem.update(entityManager);
    physicsSystem.update(entityManager);
    renderSystem.update(entityManager);
}