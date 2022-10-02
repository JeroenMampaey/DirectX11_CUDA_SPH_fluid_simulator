#pragma once

#include "event.h"
#include "mouse_move_event.h"
#include "mouse_left_click_event.h"
#include "mouse_right_click_event.h"
#include "keyboard_keydown_event.h"

#define NUM_WINDOW_EVENT_TYPES = 4

enum WindowEventType{
    MOUSE_MOVE_EVENT = 0,
    MOUSE_LEFT_CLICK_EVENT = 1,
    MOUSE_RIGHT_CLICK_EVENT = 2,
    KEYBOARD_KEYDOWN_EVENT = 3,
};