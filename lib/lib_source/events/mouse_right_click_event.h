#pragma once

#include "../event.h"

struct LIBRARY_API MouseRightClickEvent : public Event{
    int x;
    int y;
    MouseRightClickEvent(int x, int y) noexcept;
    EventType type() const noexcept override;
};