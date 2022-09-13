#include "keyboard_keydown_event.h"

KeyboardKeydownEvent::KeyboardKeydownEvent(WPARAM key) noexcept
    :
    key(key)
{}

EventType KeyboardKeydownEvent::type() const noexcept{
    return EventType::KEYBOARD_KEYDOWN_EVENT;
}