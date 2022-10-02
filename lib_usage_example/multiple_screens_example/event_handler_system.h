#pragma once

#include "entity_manager.h"

class SpecificWindowEventListener : public EventListener{
    public:
        SpecificWindowEventListener(std::shared_ptr<EventBus> windowEventBus) noexcept;
        void handleEvent(const Event& event) noexcept override;
        int entities_to_be_removed = 0;
};

class EventHandlerSystem{
    public:
        EventHandlerSystem(std::shared_ptr<EventBus> windowEventBus1, std::shared_ptr<EventBus> windowEventBus2) noexcept;
        void update(EntityManager& manager) noexcept;
    private:
        SpecificWindowEventListener window1EventListener;
        SpecificWindowEventListener window2EventListener;
};