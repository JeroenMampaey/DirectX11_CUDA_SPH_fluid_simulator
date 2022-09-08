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

#define SMOOTH 20.0
#define REST 0.2
#define STIFF 500000.0

#define PI 3.141592
#define SQRT_PI 1.772453

// M_P depends on how many particles are desired per volume unit, determined experimentally
#define M_P REST*RADIUS*RADIUS*4

#define VEL_LIMIT 800.0

#define DEBUG true
#define DEBUG_PASSED_MS_ID 0
#define DEBUG_THREAD1_ERROR 1
#define DEBUG_THREAD2_ERROR 2

#define BLOCK_SIZE 96
#define SHARED_MEM_PER_THREAD 101

#include <d2d1.h>
#include <atomic>
#include "../extra_code/console_debug.h"

typedef struct Particle{
    float x;
    float y;
} Particle;

typedef struct Boundary{
    unsigned short x1;
    unsigned short y1;
    unsigned short x2;
    unsigned short y2;
} Boundary;

typedef struct Pump{
    unsigned short x_low;
    unsigned short x_high;
    unsigned short y_low;
    unsigned short y_high;
} Pump;

typedef struct PumpVelocity{
    short velx;
    short vely;
} PumpVelocity;

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<bool> &doneDrawing, Boundary* boundaries, int numboundaries, Particle* particles, int numpoints, Pump* pumps, PumpVelocity* pumpvelocities, int numpumps, HWND m_hwnd);

#endif