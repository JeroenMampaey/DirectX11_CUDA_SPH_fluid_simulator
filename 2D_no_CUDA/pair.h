#ifndef PAIR_H
#define PAIR_H

#include "particle.h"

class Pair{
    public:
        Particle* a;
        Particle* b;

        float q;
        float q2;

        Pair(Particle* a, Particle* b){
            this->a = a;
            this->b = b;
            this->q = 0.0;
            this->q2 = 0.0;
        }
};

#endif