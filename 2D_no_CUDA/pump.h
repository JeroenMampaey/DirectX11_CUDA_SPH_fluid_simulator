#ifndef PUMP_H
#define PUMP_H

class Pump{
    public:
        float x_low;
        float x_high;
        float y_low;
        float y_high;

        float velocity_x;
        float velocity_y;

        Pump(float x_low, float x_high, float y_low, float y_high, float velocity_x, float velocity_y){
            this->x_low = x_low;
            this->x_high = x_high;
            this->y_low = y_low;
            this->y_high = y_high;
            this->velocity_x = velocity_x;
            this->velocity_y = velocity_y;
        }

        Pump(){
            Pump(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
};

#endif