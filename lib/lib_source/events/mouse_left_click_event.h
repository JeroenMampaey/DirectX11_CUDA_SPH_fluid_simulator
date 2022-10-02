#pragma once

#include "event.h"

struct LIBRARY_API MouseLeftClickEvent : public Event{
    int x;
    int y;
    MouseLeftClickEvent(int x, int y) noexcept;
    int type() const noexcept override;
};