#pragma once

#include "../event.h"

struct LIBRARY_API MouseLeftClickEvent : public Event{
    int x;
    int y;
    MouseLeftClickEvent(int x, int y) noexcept;
    EventType type() const noexcept override;
};