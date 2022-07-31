#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <vector>
#include "pair.h"

class Particle{
    public:
        float x;
        float y;
        float oldx;
        float oldy;
        float velx;
        float vely;
        float dens;
        float press;
        std::vector<Neighbor> particle_neighbors;
        std::vector<Neighbor> virtual_neighbors;

        Particle(float x, float y, float velx, float vely, float dens){
            this->x = x;
            this->y = y;
            this->oldx = x;
            this->oldy = y;
            this->velx = velx;
            this->vely = vely;
            this->dens = dens;
            this->press = 0.0;
        }

        Particle(){
            Particle(0.0, 0.0, 0.0, 0.0, 0.0);
        }
};

#endif