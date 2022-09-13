#include "mouse_move_event.h"

MouseMoveEvent::MouseMoveEvent(int new_x, int new_y) noexcept
    :
    new_x(new_x),
    new_y(new_y)
{}

EventType MouseMoveEvent::type() const noexcept{
    return EventType::MOUSE_MOVE_EVENT;
}