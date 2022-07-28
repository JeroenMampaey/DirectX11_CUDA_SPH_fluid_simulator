#include "physics.h"

void checkAllBoundaries(Particle &p, Boundary* boundaries)
{
    float boundary_tracker[] = {-1, 0, 0, 0, 0, 0};
    for(int i=0; i<NUMBOUNDARIES; i++){
        Boundary &line = boundaries[i];
        float first_check = ((p.x-line.x1)*line.nx+(p.y-line.y1)*line.ny)*((p.oldx-line.x1)*line.nx+(p.oldy-line.y1)*line.ny);
        if(first_check > 0) continue;
        float second_check1 = (line.x1-p.oldx)*line.nx+(line.y1-p.oldy)*line.ny;
        if(second_check1 < 0) continue;
        float second_check2 = (p.x-p.oldx)*line.nx+(p.y-p.oldy)*line.ny;
        float crossing_x = p.x;
        float crossing_y = p.y;
        if(second_check2 > 0.0){
            crossing_x = p.oldx+(p.x-p.oldx)*second_check1/second_check2;
            crossing_y = p.oldy+(p.y-p.oldy)*second_check1/second_check2;
        }
        float second_check3 = sqrt((crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1))/line.length;
        float second_check4 = (crossing_x-line.x1)*(line.x2-line.x1)+(crossing_y-line.y1)*(line.y2-line.y1);
        if(second_check3>1.0 || second_check4<0.0) continue;
        float adding_vector1_x = (crossing_x-p.x) - 5*line.nx;
        float adding_vector1_y = (crossing_y-p.y) - 5*line.ny;
        float adding_vector2_x = -(p.velx*line.nx+p.vely*line.ny)*line.nx-DAMPING*(p.velx*line.nx+p.vely*line.ny)*line.nx;
        float adding_vector2_y = -(p.velx*line.nx+p.vely*line.ny)*line.ny-DAMPING*(p.velx*line.nx+p.vely*line.ny)*line.ny;
        p.x += adding_vector1_x;
        p.y += adding_vector1_y;
        p.velx += adding_vector2_x;
        p.vely += adding_vector2_y;
        return;
    }
}

void updateParticles(std::atomic<int> &drawingIndex, Boundary* boundaries, Particle* particles){
    std::vector<Pair> pairs;
    for(int i=0; i<NUMPOINTS; i++){
        // Make sure the particle is not being painted on the screen at the moment
        while(drawingIndex.load() <= i){}

        Particle &p = particles[i];
        p.velx = (p.x - p.oldx) / (INTERVAL_MILI/1000.0);
        p.vely = (p.y - p.oldy) / (INTERVAL_MILI/1000.0);
        
        p.vely += GRAVITY*PIXEL_PER_METER*(INTERVAL_MILI/1000.0);

        checkAllBoundaries(p, boundaries);
        
        p.oldx = p.x;
        p.oldy = p.y;
        p.x += p.velx*(INTERVAL_MILI/1000.0);
        p.y += p.vely*(INTERVAL_MILI/1000.0);
        p.dens = 0;
        p.velx = 0;
        p.vely = 0;

        for (int j = 0; j < i; j++) {
          Particle &p2 = particles[j];
          float dist = sqrt((p.x-p2.x)*(p.x-p2.x)+(p.y-p2.y)*(p.y-p2.y));
          if (dist < SMOOTH) {
            pairs.push_back(Pair(&p, &p2));
          }
        }
    }
    drawingIndex.store(0);

    /*for (int i = 0; i < pairs.size(); i++) {
        Pair p = pairs.get(i);
        float dist = sqrt(pow(p.a.x - p.b.x, 2) + pow(p.a.y - p.b.y, 2));
        p.q = (float)(2 * Math.exp( -dist*dist / (smooth*smooth/4)) / Math.pow(smooth/2, 4) / Math.PI);
        p.q2 = (float)(Math.pow(1.0 / ((smooth/2)*Math.sqrt(Math.PI)), 2) * Math.exp( -dist*dist / (smooth*smooth/4)));
        p.a.dens += m_p*p.q2;
        p.b.dens += m_p*p.q2;
    }*/
}

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<int> &drawingIndex, Boundary* boundaries, Particle* particles, HWND m_hwnd){
    float velocity = 0;
    while(!exit.load()){
        bool expected = true;
        if(updateRequired.compare_exchange_weak(expected, false)){
            updateParticles(drawingIndex, boundaries, particles);
            InvalidateRect(m_hwnd, NULL, FALSE);
        }
    }
}