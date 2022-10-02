#include "event_handler_system.h"

SpecificWindowEventListener::SpecificWindowEventListener(std::shared_ptr<EventBus> windowEventBus) noexcept{
    subscribeTo(windowEventBus, WindowEventType::MOUSE_LEFT_CLICK_EVENT);
}

void SpecificWindowEventListener::handleEvent(const Event& event) noexcept{
    switch(event.type()){
        case WindowEventType::MOUSE_LEFT_CLICK_EVENT:
            entities_to_be_removed += 1;
            break;
    }
}

EventHandlerSystem::EventHandlerSystem(std::shared_ptr<EventBus> windowEventBus1, std::shared_ptr<EventBus> windowEventBus2) noexcept
    :
    window1EventListener(SpecificWindowEventListener(std::move(windowEventBus1))),
    window2EventListener(SpecificWindowEventListener(std::move(windowEventBus2)))
{}

void EventHandlerSystem::update(EntityManager& manager) noexcept{
    int remainingRectangles = (manager.getFilledRectangles().size() > window1EventListener.entities_to_be_removed) ? manager.getFilledRectangles().size()-window1EventListener.entities_to_be_removed : 0;
    manager.getFilledRectangles().erase(std::next(manager.getFilledRectangles().begin(), remainingRectangles), manager.getFilledRectangles().end());
    window1EventListener.entities_to_be_removed = 0;

    int remainingCircles = (manager.getCircles().size() > window2EventListener.entities_to_be_removed) ? manager.getCircles().size()-window2EventListener.entities_to_be_removed : 0;
    manager.getCircles().erase(std::next(manager.getCircles().begin(), remainingCircles), manager.getCircles().end());
    window2EventListener.entities_to_be_removed = 0;
}