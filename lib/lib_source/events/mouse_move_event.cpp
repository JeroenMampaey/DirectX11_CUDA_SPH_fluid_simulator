#include "mouse_move_event.h"
#include "events_includes.h"

MouseMoveEvent::MouseMoveEvent(int new_x, int new_y) noexcept
    :
    new_x(new_x),
    new_y(new_y)
{}

int MouseMoveEvent::type() const noexcept{
    return WindowEventType::MOUSE_MOVE_EVENT;
}