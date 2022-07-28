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

#define DAMPING 0.1

#define SMOOTH 35

#include <d2d1.h>
#include <atomic>
#include "boundary.h"
#include "particle.h"
#include "pair.h"
#include <vector>

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<int> &drawingIndex, Boundary* boundaries, Particle* particles, HWND m_hwnd);

#endif