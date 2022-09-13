#include "mouse_right_click_event.h"

MouseRightClickEvent::MouseRightClickEvent(int x, int y) noexcept
    :
    x(x),
    y(y)
{}

EventType MouseRightClickEvent::type() const noexcept{
    return EventType::MOUSE_RIGHT_CLICK_EVENT;
}