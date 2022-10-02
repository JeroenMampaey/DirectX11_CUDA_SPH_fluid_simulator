#pragma once

#include "entity_manager.h"

class EventHandlerSystem : public EventListener{
    public:
        EventHandlerSystem(std::shared_ptr<EventBus> windowEventBus) noexcept;
        void handleEvent(const Event& event) noexcept override;
        void update(EntityManager& manager) noexcept;
    private:
        void handleKeyEvent(LPARAM key) noexcept;
        float circle_displacement = 0.0f;
        float camera_displacement = 0.0f;
        int circles_to_be_removed = 0;
        bool updateTextField = false;
};