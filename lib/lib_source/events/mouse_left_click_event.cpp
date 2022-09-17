#include "mouse_left_click_event.h"
#include "events_includes.h"

MouseLeftClickEvent::MouseLeftClickEvent(int x, int y) noexcept
    :
    x(x),
    y(y)
{}

int MouseLeftClickEvent::type() const noexcept{
    return WindowEventType::MOUSE_LEFT_CLICK_EVENT;
}