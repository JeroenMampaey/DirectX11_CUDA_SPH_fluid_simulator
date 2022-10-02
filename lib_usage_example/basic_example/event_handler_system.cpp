#include "event_handler_system.h"

EventHandlerSystem::EventHandlerSystem(std::shared_ptr<EventBus> windowEventBus) noexcept{
    subscribeTo(windowEventBus, WindowEventType::MOUSE_LEFT_CLICK_EVENT);
    subscribeTo(windowEventBus, WindowEventType::MOUSE_RIGHT_CLICK_EVENT);
    subscribeTo(windowEventBus, WindowEventType::KEYBOARD_KEYDOWN_EVENT);
}

void EventHandlerSystem::handleEvent(const Event& event) noexcept{
    switch(event.type()){
        case WindowEventType::MOUSE_LEFT_CLICK_EVENT:
            circle_displacement += 50.0f;
            break;
        case WindowEventType::MOUSE_RIGHT_CLICK_EVENT:
            circle_displacement -= 50.0f;
            break;
        case WindowEventType::KEYBOARD_KEYDOWN_EVENT:
            {
                const KeyboardKeydownEvent& castedEvent = static_cast<const KeyboardKeydownEvent&>(event);
                handleKeyEvent(castedEvent.key);
            }
            break;
    }
}

void EventHandlerSystem::update(EntityManager& manager) noexcept{
    int remainingCircles = (manager.getCircles().size() > circles_to_be_removed) ? manager.getCircles().size()-circles_to_be_removed : 0;
    manager.getCircles().erase(std::next(manager.getCircles().begin(), remainingCircles), manager.getCircles().end());
    circles_to_be_removed = 0;

    for(CircleEntity& circle : manager.getCircles()){
        circle.y += circle_displacement;
    }
    circle_displacement = 0.0f;

    manager.getCamera().x += camera_displacement;
    camera_displacement = 0.0f;

    if(updateTextField){
        manager.getSpecificTextField().text += std::to_string(manager.getSpecificTextField().counter);
        manager.getSpecificTextField().counter++;
    }
    updateTextField = false;
}

void EventHandlerSystem::handleKeyEvent(LPARAM key) noexcept{
    switch(key){
        case 'A':
            circles_to_be_removed++;
            break;
        case 'B':
            updateTextField = true;
            break;
        case VK_LEFT:
            camera_displacement -= 30.0f;
            break;
        case VK_RIGHT:
            camera_displacement += 30.0f;
            break;
    }
}