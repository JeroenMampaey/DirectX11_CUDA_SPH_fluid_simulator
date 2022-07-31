#ifndef PAIR_H
#define PAIR_H

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

class BoundaryPair{
    public:
        Particle* a;
        float boundary_x;
        float boundary_y;
        float boundary_mass;

        float q;
        float q2;

        float dist;

        BoundaryPair(Particle* a, float boundary_x, float boundary_y, float boundary_mass, float dist){
            this->a = a;
            this->boundary_x = boundary_x;
            this->boundary_y = boundary_y;
            this->boundary_mass = boundary_mass;

            this->q = 0.0;
            this->q2 = 0.0;

            this->dist = dist;
        }
};

#endif