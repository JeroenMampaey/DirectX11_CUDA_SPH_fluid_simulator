#include "physics.h"
#include <vector>
#include <string>

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<bool> &doneDrawing, Boundary* boundaries, int numboundaries, Particle* particles, int numpoints, Pump* pumps, PumpVelocity* pumpvelocities, int numpumps, HWND m_hwnd){
    while(!exit.load()){
        bool expected = true;
        if(updateRequired.compare_exchange_weak(expected, false)){
            // If an update is necessary, update the particles UPDATES_PER_RENDER times and then redraw the particles
            //TODO
            doneDrawing.store(false);

            // Redraw the particles
            InvalidateRect(m_hwnd, NULL, FALSE);
        }
    }
}