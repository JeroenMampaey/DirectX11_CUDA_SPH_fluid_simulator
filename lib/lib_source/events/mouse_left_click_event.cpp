#include "mouse_left_click_event.h"

MouseLeftClickEvent::MouseLeftClickEvent(int x, int y) noexcept
    :
    x(x),
    y(y)
{}

EventType MouseLeftClickEvent::type() const noexcept{
    return EventType::MOUSE_LEFT_CLICK_EVENT;
}