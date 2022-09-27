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

class Point{
    public:
        float x;
        float y;
        Point(float x, float y) noexcept;
};

class Vector{
    public:
        float x;
        float y;
        Vector(float x, float y) noexcept;
};

class Boundary{
    public:
        const float lengthSquared;
        
        const Point point1;
        const Point point2;

        const Vector normal;
        const Vector direction;

        Boundary(float x1, float y1, float x2, float y2);
};

class Particle;

class Neighbour{
    public:
        const Particle& p;

        const float kernel;
        const float gradKernel;

        Neighbour(const Particle& p, float kernel, float gradKernel) noexcept;
};

class VirtualNeighbour : public Neighbour{
    public:
        const float virtualX;
        const float virtualY;
        VirtualNeighbour(const Particle& p, float kernel, float gradKernel, float virtualX, float virtualY) noexcept;
};

class Particle{
    public:
        Point pos;
        Point oldPos;

        Vector vel;
        
        float dens;
        float pressure_density_ratio;

        std::vector<Neighbour> neighbours;
        std::vector<std::pair<std::vector<VirtualNeighbour>, const Boundary&>> virtualNeighbours;

        Particle(float x, float y) noexcept;
};

class Pump{
    public:
        const Point leftBottom;
        const Point rightTop;

        const Vector vel;

        Pump(float xLow, float xHigh, float yLow, float yHigh, float velocityX, float velocityY) noexcept;
};

class EntityManager{
    public:
        EntityManager();

        std::vector<Particle>& getParticles() noexcept;
        std::vector<Boundary>& getBoundaries() noexcept;
        std::vector<Pump>& getPumps() noexcept;
    
    private:
        void buildDefaultSimulationLayout();
        void buildSimulationLayoutFromFile(char* buffer);

        std::vector<Boundary> boundaries;
        std::vector<Pump> pumps;
        std::vector<Particle> particles;
};