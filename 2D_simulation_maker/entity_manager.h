#pragma once

#include <vector>
#include "../lib/lib_header.h"

#define DEFAULT_PUMP_VELOCITY 200.0f

class Particle{
    public:
        float x;
        float y;
        Particle(float x, float y);
};

class Boundary{
    public:
        float x1;
        float y1;
        float x2;
        float y2;
        Boundary(float x1, float y1, float x2, float y2);
};

class Pump{
    public:
        float x1;
        float y1;
        float x2;
        float y2;
        float vel_x;
        float vel_y;
        Pump(float x1, float y1, float x2, float y2, float vel_x, float vel_y);
        Pump(float x1, float y1, float x2, float y2);
};

class EntityManager{
    public:
        std::vector<Particle>& getParticles() noexcept;
        std::vector<Boundary>& getBoundaries() noexcept;
        std::vector<Pump>& getPumps() noexcept;
    private:
        std::vector<Particle> particles;
        std::vector<Boundary> boundaries;
        std::vector<Pump> pumps;
};