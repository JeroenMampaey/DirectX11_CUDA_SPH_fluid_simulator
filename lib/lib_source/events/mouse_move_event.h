#pragma once

#include "event.h"

struct LIBRARY_API MouseMoveEvent : public Event{
    int new_x;
    int new_y;
    MouseMoveEvent(int new_x, int new_y) noexcept;
    int type() const noexcept override;
};