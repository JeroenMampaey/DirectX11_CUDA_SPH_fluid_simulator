#ifndef PHYSICS_H
#define PHYSICS_H

#define CEILING 0
#define FLOOR 700
#define RIGHT 1280
#define LEFT 0
#define RADIUS 7.5
#define INTERVAL_MILI 10.0
#define GRAVITY 9.8
#define PIXEL_PER_METER 100.0

#define NUMPOINTS 1000
#define NUMBOUNDARIES 3

#include <d2d1.h>
#include <atomic>
#include "boundary.h"
#include "particle.h"

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, int* height, HWND m_hwnd);

#endif