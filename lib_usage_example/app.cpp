#include "app.h"

#define SYNCINTERVAL 1

App::App()
    :
    wnd(createWindow("Example", SYNCINTERVAL)),
    entityManager(EntityManager(wnd.getGraphicsEngine())),
    eventHandlerSystem(EventHandlerSystem(wnd.getEventBus())),
    renderSystem(RenderSystem(wnd.getGraphicsEngine(), entityManager))
{
    if(RATE_IS_INVALID(wnd.getGraphicsEngine().getRefreshRate())){
        throw std::exception("Refreshrate could not easily be found programmatically.");
    }
    float dt = 1.0f/wnd.getGraphicsEngine().getRefreshRate();
    physicsSystem = PhysicsSystem(dt);
}

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