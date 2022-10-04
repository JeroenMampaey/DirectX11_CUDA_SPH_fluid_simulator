#include "app.h"

#define SYNCINTERVAL 1

App::App()
    :
    wnd(createWindow("Example", SYNCINTERVAL)),
    entityManager(EntityManager(wnd.getGraphicsEngine())),
    physicsSystem(PhysicsSystem(wnd.getGraphicsEngine(), entityManager)),
    renderSystem(RenderSystem(wnd.getGraphicsEngine()))
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