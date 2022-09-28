#pragma once

#include "../lib/lib_header.h"
#include <vector>
#include <utility>
#include <cmath>

#ifndef SIMULATION_LAYOUT_DIRECTORY
#define SIMULATION_LAYOUT_DIRECTORY "../../simulation_layout/"
#endif

#define SLD_PATH_CONCATINATED(original) SIMULATION_LAYOUT_DIRECTORY original

#define RADIUS 7.5f

struct Particle{
    float x;
    float y;
};

struct Boundary{
    unsigned short x1;
    unsigned short y1;
    unsigned short x2;
    unsigned short y2;
};

struct Pump{
    unsigned short xLow;
    unsigned short xHigh;
    unsigned short yLow;
    unsigned short yHigh;
};

struct PumpVelocity{
    short velX;
    short velY;
};

class EntityManager{
    public:
        EntityManager();

        std::vector<Particle>& getParticles() noexcept;
        std::vector<Boundary>& getBoundaries() noexcept;
        std::vector<Pump>& getPumps() noexcept;
        std::vector<PumpVelocity>& getPumpVelocities() noexcept;
    
    private:
        void buildDefaultSimulationLayout();
        void buildSimulationLayoutFromFile(char* buffer);

        std::vector<Boundary> boundaries;
        std::vector<Pump> pumps;
        std::vector<Particle> particles;
        std::vector<PumpVelocity> pumpVelocities;
};