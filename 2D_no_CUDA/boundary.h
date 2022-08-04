#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <cmath>

class Boundary{
    public:
        float x1;
        float y1;
        float x2;
        float y2;
        float nx;
        float ny;
        float length_squared;
        float px;
        float py;

        Boundary(float x1, float y1, float x2, float y2){
            this->x1 = x1;
            this->y1 = y1;
            this->x2 = x2;
            this->y2 = y2;
            this->length_squared = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
            float length = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
            this->nx = (y2-y1)/length;
            this->ny = (x1-x2)/length;
            this->px = (x2-x1)/length;
            this->py = (y2-y1)/length;
        }

        Boundary(){
            Boundary(0.0, 0.0, 1.0, 0.0);
        }
};

#endif