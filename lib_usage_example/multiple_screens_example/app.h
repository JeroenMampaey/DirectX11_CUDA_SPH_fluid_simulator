#pragma once

#include "render_system.h"
#include "event_handler_system.h"

class App{
    public:
        App();
        int go();
    
    private:
        void updateSystems();

        Window wnd1;
        Window wnd2;
        EntityManager entityManager;
        RenderSystem renderSystem;
        EventHandlerSystem eventHandlerSystem;
};