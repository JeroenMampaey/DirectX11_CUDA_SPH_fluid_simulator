#include "app.h"

#define SYNCINTERVAL 1

App::App()
    :
    wnd(createWindow("Example", SYNCINTERVAL)),
    entityManager(EntityManager()),
    physicsSystem(PhysicsSystem(wnd.getGraphicsEngine())),
    renderSystem(RenderSystem(wnd.getGraphicsEngine(), entityManager))
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
    physicsSystem.update(entityManager);
    renderSystem.update(entityManager);
}