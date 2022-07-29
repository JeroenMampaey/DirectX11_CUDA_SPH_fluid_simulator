#ifndef PHYSICS_H
#define PHYSICS_H

#define CEILING 0
#define FLOOR 700
#define RIGHT 1280
#define LEFT 0
#define RADIUS 7.5
#define INTERVAL_MILI 5.0
#define UPDATES_PER_RENDER 6
#define GRAVITY 9.8
#define PIXEL_PER_METER 100.0

#define DAMPING 0.1

#define SMOOTH 35.0
#define REST 0.2
#define MULTIPLIER 6.0
#define STIFF 130000.0

#define PI 3.141592
#define SQRT_PI 1.772453

// M_P depends on how many particles are desired per volume unit, determined experimentally
#define M_P REST*RADIUS*RADIUS*4*3

#include <d2d1.h>
#include <atomic>
#include "boundary.h"
#include "particle.h"

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<int> &drawingIndex, Boundary* boundaries, int numboundaries, Particle* particles, int numpoints, HWND m_hwnd);

#endif