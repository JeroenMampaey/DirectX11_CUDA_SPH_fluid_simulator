#include "physics.h"
#include "pair.h"
#include <vector>

void addBoundaryParticles(Particle &p, Boundary &line){
    if(p.particle_neighbors.size()<4) return;
    for(Neighbor neigbor : p.particle_neighbors){
        float projection = (line.x1 - neigbor.x)*line.nx + (line.y1 - neigbor.y)*line.ny;
        float virtual_x = neigbor.x + 2*projection*line.nx;
        float virtual_y = neigbor.y + 2*projection*line.ny;
        float second_check1 = (line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny;
        float second_check2 = (virtual_x-p.x)*line.nx+(virtual_y-p.y)*line.ny;
        float crossing_x = p.x;
        float crossing_y = p.y;
        if(second_check2 > 0.0){
            crossing_x = p.x+(virtual_x-p.x)*second_check1/second_check2;
            crossing_y = p.y+(virtual_y-p.y)*second_check1/second_check2;
        }
        float second_check3 = sqrt((crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1))/line.length;
        float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
        if(second_check3>1.0 || second_check4<0.0) continue;
        float dist = sqrt((virtual_x-p.x)*(virtual_x-p.x)+(virtual_y-p.y)*(virtual_y-p.y));
        if(dist > SMOOTH) continue;
        float q = (float)(2*exp( -dist*dist / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
        float q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist*dist / (SMOOTH*SMOOTH/4)));
        p.dens += M_P*q2;
        p.virtual_neighbors.push_back(Neighbor(neigbor.p, dist, virtual_x, virtual_y, q, q2));
    }
    float projection = (line.x1 - p.x)*line.nx + (line.y1 - p.y)*line.ny;
    float virtual_x = p.x + 2*projection*line.nx;
    float virtual_y = p.y + 2*projection*line.ny;
    float second_check1 = (line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny;
    float second_check2 = (virtual_x-p.x)*line.nx+(virtual_y-p.y)*line.ny;
    float crossing_x = p.x;
    float crossing_y = p.y;
    if(second_check2 > 0.0){
        crossing_x = p.x+(virtual_x-p.x)*second_check1/second_check2;
        crossing_y = p.y+(virtual_y-p.y)*second_check1/second_check2;
    }
    float second_check3 = sqrt((crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1))/line.length;
    float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
    if(second_check3>1.0 || second_check4<0.0) return;
    float dist = sqrt((virtual_x-p.x)*(virtual_x-p.x)+(virtual_y-p.y)*(virtual_y-p.y));
    if(dist > SMOOTH) return;
    float q = (float)(2*exp( -dist*dist / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
    float q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist*dist / (SMOOTH*SMOOTH/4)));
    p.dens += M_P*q2;
    p.virtual_neighbors.push_back(Neighbor(&p, dist, virtual_x, virtual_y, q, q2));

}

// Loop over all boundaries and check if Particle p has crossed a boundary and if so, 
// change the position and velocity of p as to simulate a collision according (or atleast somewhat similar to) 
// regular classical mechanics
void checkAllBoundaries(Particle &p, Boundary* boundaries, int numboundaries){
    for(int i=0; i<numboundaries; i++){
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
        float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
        if(second_check3>1.0 || second_check4<0.0) continue;
        p.x = crossing_x - 5*line.nx;
        p.y = crossing_y - 5*line.ny;
        p.velx += -(p.velx*line.nx+p.vely*line.ny)*line.nx-DAMPING*(p.velx*line.nx+p.vely*line.ny)*line.nx;
        p.vely += -(p.velx*line.nx+p.vely*line.ny)*line.ny-DAMPING*(p.velx*line.nx+p.vely*line.ny)*line.ny;
        return;
    }
}

void updateParticles(std::atomic<int> &drawingIndex, Boundary* boundaries, int numboundaries, Particle* particles, int numpoints){
    std::vector<Pair> pairs;
    std::vector<BoundaryPair> boundary_pairs;

    for(int i=0; i<numpoints; i++){
        // Make sure the particle is not being painted on the screen at the moment
        while(drawingIndex.load() <= i){}

        Particle &p = particles[i];
        p.velx = (p.x - p.oldx) / (INTERVAL_MILI/1000.0);
        p.vely = (p.y - p.oldy) / (INTERVAL_MILI/1000.0);
        
        // Update positional change of particles caused by gravity
        p.vely += GRAVITY*PIXEL_PER_METER*(INTERVAL_MILI/1000.0);

        // Update positional change of particles caused boundaries (make sure particles cannot pass boundaries)
        checkAllBoundaries(p, boundaries, numboundaries);
        
        p.oldx = p.x;
        p.oldy = p.y;
        p.x += p.velx*(INTERVAL_MILI/1000.0);
        p.y += p.vely*(INTERVAL_MILI/1000.0);
        p.dens = 0;
        p.velx = 0;
        p.vely = 0;
        p.particle_neighbors.clear();
        p.virtual_neighbors.clear();

        // Find particles that are near to each other and calculate densities
        for (int j = 0; j < i; j++) {
          Particle &p2 = particles[j];
          float dist = sqrt((p.x-p2.x)*(p.x-p2.x)+(p.y-p2.y)*(p.y-p2.y));
          if (dist < SMOOTH) {
            float q = (float)(2*exp( -dist*dist / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
            float q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist*dist / (SMOOTH*SMOOTH/4)));
            p.dens += M_P*q2;
            p2.dens += M_P*q2;
            p.particle_neighbors.push_back(Neighbor(&p2, dist, p2.x, p2.y, q, q2));
            p2.particle_neighbors.push_back(Neighbor(&p, dist, p.x, p.y, q, q2));
          }
        }
    }

    for(int i = 0; i<numpoints; i++){
        Particle &p = particles[i];
        for(int j=0; j<numboundaries; j++){
            Boundary &line = boundaries[j];
            float projection = (line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny;
            float crossing_x = p.x + projection*line.nx;
            float crossing_y = p.y + projection*line.ny;
            float second_check3 = sqrt((crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1))/line.length;
            float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
            if(projection <= SMOOTH && projection >= 0 && second_check3 <= 1 && second_check4 >= 0){
                addBoundaryParticles(p, line);
            }
            else if(sqrt((line.x1-p.x)*(line.x1-p.x) + (line.y1-p.y)*(line.y1-p.y)) < SMOOTH && ((line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny) > 0 && projection>=0){
                addBoundaryParticles(p, line);
            }
            else if(sqrt((line.x2-p.x)*(line.x2-p.x) + (line.y2-p.y)*(line.y2-p.y)) < SMOOTH && ((line.x2-p.x)*line.nx+(line.y2-p.y)*line.ny) > 0 && projection>=0){
                addBoundaryParticles(p, line);
            }
        }
    }
    
    // Calculate the pressure of each particle based on their density
    for (int i = 0; i < numpoints; i++) {
        Particle &p = particles[i];
        p.press = STIFF * (p.dens - REST);
    }

    for(int i = 0; i<numpoints; i++){
        Particle &p = particles[i]; 
        for(Neighbor neighbor : p.particle_neighbors){
            float press = M_P*(p.press/(p.dens*p.dens) + neighbor.p->press/(neighbor.p->dens*neighbor.p->dens));
            float displace = (press * neighbor.q) * (INTERVAL_MILI/1000.0);
            float abx = (p.x - neighbor.x);
            float aby = (p.y - neighbor.y);
            p.velx += displace * abx;
            p.vely += displace * aby;
        }

        for(Neighbor neighbor : p.virtual_neighbors){
            float press = M_P*(p.press/(p.dens*p.dens) + neighbor.p->press/(neighbor.p->dens*neighbor.p->dens));
            float displace = (press * neighbor.q) * (INTERVAL_MILI/1000.0);
            float abx = (p.x - neighbor.x);
            float aby = (p.y - neighbor.y);
            p.velx += displace * abx;
            p.vely += displace * aby;
        }
    }
    
    // Update the position of particles based on Euler's equation for an ideal fluid
    for (int i = 0; i < numpoints; i++) {
        Particle &p = particles[i];
        p.x += p.velx * (INTERVAL_MILI/1000.0);
        p.y += p.vely * (INTERVAL_MILI/1000.0);
    }
}

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<int> &drawingIndex, Boundary* boundaries, int numboundaries, Particle* particles, int numpoints, HWND m_hwnd){
    float velocity = 0;
    while(!exit.load()){
        bool expected = true;
        if(updateRequired.compare_exchange_weak(expected, false)){
            // If an update is necessary, update the particles UPDATES_PER_RENDER times and then redraw the particles
            for(int i=0; i<UPDATES_PER_RENDER; i++) updateParticles(drawingIndex, boundaries, numboundaries, particles, numpoints);
            drawingIndex.store(0);

            // Redraw the particles
            InvalidateRect(m_hwnd, NULL, FALSE);
        }
    }
}