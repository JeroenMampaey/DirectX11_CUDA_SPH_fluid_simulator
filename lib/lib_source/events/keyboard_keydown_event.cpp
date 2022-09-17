#include "keyboard_keydown_event.h"
#include "events_includes.h"

KeyboardKeydownEvent::KeyboardKeydownEvent(WPARAM key) noexcept
    :
    key(key)
{}

int KeyboardKeydownEvent::type() const noexcept{
    return WindowEventType::KEYBOARD_KEYDOWN_EVENT;
}