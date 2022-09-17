#pragma once

#include "line.h"
#include "filled_circle.h"
#include "filled_rectangle.h"
#include "hollow_rectangle.h"
#include "text.h"
#include "screen_text.h"

enum DrawableType{
    LINE,
    FILLED_CIRCLE,
    FILLED_RECTANGLE,
    HOLLOW_RECTANGLE,
    TEXT,
    SCREEN_TEXT,
    Count
};