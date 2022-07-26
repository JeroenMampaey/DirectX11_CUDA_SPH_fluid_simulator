#include "physics.h"

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, int* height, HWND m_hwnd){
    float velocity = 0;
    while(!exit.load()){
        bool expected = true;
        if(updateRequired.compare_exchange_weak(expected, false)){
            velocity += GRAVITY*INTERVAL_MILI/1000;
            *height += (int)(velocity*PIXEL_PER_METER*INTERVAL_MILI/1000);
            InvalidateRect(m_hwnd, NULL, FALSE);
        }
    }
}