#pragma once

#include "line.h"
#include "filled_circle.h"
#include "filled_rectangle.h"
#include "hollow_rectangle.h"
#include "text.h"
#include "screen_text.h"

enum DrawableType{
    LINE = 0,
    FILLED_CIRCLE = 1,
    FILLED_RECTANGLE = 2,
    HOLLOW_RECTANGLE = 3,
    TEXT = 4,
    SCREEN_TEXT = 5
};