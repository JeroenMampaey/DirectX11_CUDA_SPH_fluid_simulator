#ifndef PAIR_H
#define PAIR_H

#include <vector>

class Particle;

class Neighbor{
    public:
        Particle* p;
        float dist;
        float x;
        float y;

        float q;
        float q2;

        Neighbor(Particle* p, float dist, float x, float y, float q, float q2){
            this->p = p;
            this->dist = dist;
            this->x = x;
            this->y = y;
            this->q = q;
            this->q2 = q2;
        }
};

class VirtualNeigbors{
    public:
        std::vector<Neighbor> neighbors;
        float boundary_nx;
        float boundary_ny;

        VirtualNeigbors(float boundary_nx, float boundary_ny){
            this->boundary_nx = boundary_nx;
            this->boundary_ny = boundary_ny;
        }
};

#endif