#pragma once

#include <vector>
#include "../lib/lib_header.h"

#define DEFAULT_PUMP_VELOCITY 2.0f
#define PARTICLE_ZONE_RADIUS 30.0f

class ParticleZone{
    public:
        float x;
        float y;
        ParticleZone(float x, float y);
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
        std::vector<ParticleZone>& getParticleZones() noexcept;
        std::vector<Boundary>& getBoundaries() noexcept;
        std::vector<Pump>& getPumps() noexcept;
    private:
        std::vector<ParticleZone> particleZones;
        std::vector<Boundary> boundaries;
        std::vector<Pump> pumps;
};