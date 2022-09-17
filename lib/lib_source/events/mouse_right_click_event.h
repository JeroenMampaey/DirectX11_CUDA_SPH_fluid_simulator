#pragma once

#include "../event.h"

struct LIBRARY_API MouseRightClickEvent : public Event{
    int x;
    int y;
    MouseRightClickEvent(int x, int y) noexcept;
    int type() const noexcept override;
};