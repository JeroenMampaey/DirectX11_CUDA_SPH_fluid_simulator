#pragma once

#include "event.h"
#include "../windows_includes.h"

struct LIBRARY_API KeyboardKeydownEvent : public Event{
    WPARAM key;
    KeyboardKeydownEvent(WPARAM key) noexcept;
    int type() const noexcept override;
};