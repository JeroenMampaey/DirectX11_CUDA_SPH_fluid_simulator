#pragma once

#include "physics_system.h"
#include "render_system.h"

class App{
    public:
        App();
        int go();
    
    private:
        void updateSystems();

        Window wnd;
        EntityManager entityManager;
        PhysicsSystem physicsSystem;
        RenderSystem renderSystem;
};