#pragma once

#include "../lib/lib_header.h"

#pragma once

#include "render_system.h"
#include "event_handler_system.h"

class App{
    public:
        App();
        int go();
    
    private:
        Window wnd;
        EntityManager entityManager;
        RenderSystem renderSystem;
        EventHandlerSystem eventHandlerSystem;
};