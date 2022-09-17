#include "mouse_right_click_event.h"
#include "events_includes.h"

MouseRightClickEvent::MouseRightClickEvent(int x, int y) noexcept
    :
    x(x),
    y(y)
{}

int MouseRightClickEvent::type() const noexcept{
    return WindowEventType::MOUSE_RIGHT_CLICK_EVENT;
}