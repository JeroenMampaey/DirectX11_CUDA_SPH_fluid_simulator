#include "app.h"

// a SYNCINTERVAL that provides FPS close to 30 is optimal when processing the physics using CPU only
#define SYNCINTERVAL 4

App::App()
    :
    wnd(Window("Example", SYNCINTERVAL)),
    entityManager(EntityManager()),
    physicsSystem(PhysicsSystem(wnd.getGraphicsEngine())),
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