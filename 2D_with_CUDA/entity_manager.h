#pragma once

#include "../lib/lib_header.h"
#include <vector>
#include <utility>
#include <cmath>
#include "exceptions.h"

#ifndef SIMULATION_LAYOUT_DIRECTORY
#define SIMULATION_LAYOUT_DIRECTORY "../../simulation_layout/"
#endif

#define SLD_PATH_CONCATINATED(original) SIMULATION_LAYOUT_DIRECTORY original

#define RADIUS 7.5f

#define Particle DirectX::XMFLOAT4

#define MAX_POSSIBLE_PARTICLES 23040

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
        EntityManager(GraphicsEngine& gfx);

        CudaAccessibleFilledCircleInstanceBuffer& getParticles() noexcept;
        std::vector<Boundary>& getBoundaries() noexcept;
        std::vector<Pump>& getPumps() noexcept;
        std::vector<PumpVelocity>& getPumpVelocities() noexcept;
    
    private:
        void buildDefaultSimulationLayout(GraphicsEngine& gfx);
        void buildSimulationLayoutFromFile(GraphicsEngine& gfx, char* buffer);

        std::vector<Boundary> boundaries;
        std::vector<Pump> pumps;
        std::unique_ptr<CudaAccessibleFilledCircleInstanceBuffer> pParticles;
        std::vector<PumpVelocity> pumpVelocities;
};