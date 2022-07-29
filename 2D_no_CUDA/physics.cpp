#include "physics.h"
#include "pair.h"
#include <vector>

// Add a virtual particle that simulates density effects caused by a corner of two boundary lines for Particle p
void addCornerBoundaryParticle(float crossing_x, float crossing_y, float distance, Particle &p, std::vector<BoundaryPair> &boundary_pairs){
    float direction_x = (crossing_x-p.x)/distance;
    float direction_y = (crossing_y-p.y)/distance;
    float x = crossing_x+direction_x*(SMOOTH/2);
    float y = crossing_y+direction_y*(SMOOTH/2);
    float mass = MULTIPLIER*((p.dens < REST) ? REST : p.dens)*SMOOTH*SMOOTH/8;
    boundary_pairs.push_back(BoundaryPair(&p, x, y, mass, distance+SMOOTH/2));
}

// Add virtual particles along Boundary line to try to simulate density effects of a real boundary for Particle p 
void addBoundaryParticles(float crossing_x, float crossing_y, Boundary &line, Particle &p, std::vector<BoundaryPair> &boundary_pairs){
    float distance_to_1 = sqrt((line.x1-crossing_x)*(line.x1-crossing_x)+(line.y1-crossing_y)*(line.y1-crossing_y));
    float distance_to_2 = sqrt((line.x2-crossing_x)*(line.x2-crossing_x)+(line.y2-crossing_y)*(line.y2-crossing_y));
    float distance;
    if((distance = sqrt((crossing_x+(SMOOTH/2)*line.nx-p.x)*(crossing_x+(SMOOTH/2)*line.nx-p.x)+(crossing_y+(SMOOTH/2)*line.ny-p.y)*(crossing_y+(SMOOTH/2)*line.ny-p.y)))<SMOOTH){
      float x = crossing_x+(SMOOTH/2)*line.nx;
      float y = crossing_y+(SMOOTH/2)*line.ny;
      float mass = MULTIPLIER*((p.dens < REST) ? REST : p.dens)*SMOOTH*SMOOTH/10;
      boundary_pairs.push_back(BoundaryPair(&p, x, y, mass, distance));
    }
    for(float i = SMOOTH/10; i<distance_to_1 && (distance=sqrt((crossing_x+(SMOOTH/2)*line.nx-line.px*i-p.x)*(crossing_x+(SMOOTH/2)*line.nx-line.px*i-p.x)+(crossing_y+(SMOOTH/2)*line.ny-line.py*i-p.y)*(crossing_y+(SMOOTH/2)*line.ny-line.py*i-p.y)))<SMOOTH; i += SMOOTH/10){
      float x = crossing_x+(SMOOTH/2)*line.nx-line.px*i;
      float y = crossing_y+(SMOOTH/2)*line.ny-line.py*i;
      float mass = MULTIPLIER*((p.dens < REST) ? REST : p.dens)*SMOOTH*SMOOTH/10;
      boundary_pairs.push_back(BoundaryPair(&p, x, y, mass, distance));
    }
    for(float i = SMOOTH/10; i<distance_to_2 && (distance=sqrt((crossing_x+(SMOOTH/2)*line.nx+line.px*i-p.x)*(crossing_x+(SMOOTH/2)*line.nx+line.px*i-p.x)+(crossing_y+(SMOOTH/2)*line.ny+line.py*i-p.y)*(crossing_y+(SMOOTH/2)*line.ny+line.py*i-p.y)))<SMOOTH; i += SMOOTH/10){
      float x = crossing_x+(SMOOTH/2)*line.nx+line.px*i;
      float y = crossing_y+(SMOOTH/2)*line.ny+line.py*i;
      float mass = MULTIPLIER*((p.dens < REST) ? REST : p.dens)*SMOOTH*SMOOTH/10;
      boundary_pairs.push_back(BoundaryPair(&p, x, y, mass, distance));
    }
}

// Loop over all boundaries and check if Particle p has crossed a boundary and if so, 
// change the position and velocity of p as to simulate a collision according (or atleast somewhat similar to) 
// regular classical mechanics
void checkAllBoundaries(Particle &p, Boundary* boundaries){
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
        float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
        if(second_check3>1.0 || second_check4<0.0) continue;
        p.x = crossing_x - 5*line.nx;
        p.y = crossing_y - 5*line.ny;
        p.velx += -(p.velx*line.nx+p.vely*line.ny)*line.nx-DAMPING*(p.velx*line.nx+p.vely*line.ny)*line.nx;
        p.vely += -(p.velx*line.nx+p.vely*line.ny)*line.ny-DAMPING*(p.velx*line.nx+p.vely*line.ny)*line.ny;
        return;
    }
}

void updateParticles(std::atomic<int> &drawingIndex, Boundary* boundaries, Particle* particles){
    std::vector<Pair> pairs;
    std::vector<BoundaryPair> boundary_pairs;

    for(int i=0; i<NUMPOINTS; i++){
        // Make sure the particle is not being painted on the screen at the moment
        while(drawingIndex.load() <= i){}

        Particle &p = particles[i];
        p.velx = (p.x - p.oldx) / (INTERVAL_MILI/1000.0);
        p.vely = (p.y - p.oldy) / (INTERVAL_MILI/1000.0);
        
        // Update positional change of particles caused by gravity
        p.vely += GRAVITY*PIXEL_PER_METER*(INTERVAL_MILI/1000.0);

        // Update positional change of particles caused boundaries (make sure particles cannot pass boundaries)
        checkAllBoundaries(p, boundaries);
        
        p.oldx = p.x;
        p.oldy = p.y;
        p.x += p.velx*(INTERVAL_MILI/1000.0);
        p.y += p.vely*(INTERVAL_MILI/1000.0);
        p.dens = 0;
        p.velx = 0;
        p.vely = 0;

        // Find particles that are near to each other and calculate densities
        for (int j = 0; j < i; j++) {
          Particle &p2 = particles[j];
          float dist = sqrt((p.x-p2.x)*(p.x-p2.x)+(p.y-p2.y)*(p.y-p2.y));
          if (dist < SMOOTH) {
            Pair new_pair = Pair(&p, &p2);
            new_pair.q = (float)(2*exp( -dist*dist / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
            new_pair.q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist*dist / (SMOOTH*SMOOTH/4)));
            pairs.push_back(new_pair);
            p.dens += M_P*new_pair.q2;
            p2.dens += M_P*new_pair.q2;
          }
        }
    }

    for(int i=0; i<NUMPOINTS; i++){
        Particle &p = particles[i];
        // Find boundaries that are near enough to particles to possibly influence their density
        for(int j=0; j<NUMBOUNDARIES; j++){
            Boundary &line = boundaries[j];
            float projection = (line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny;
            float crossing_x = p.x + projection*line.nx;
            float crossing_y = p.y + projection*line.ny;
            float second_check3 = sqrt((crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1))/line.length;
            float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
            float distance;
            if(projection <= SMOOTH/2 && projection >= 0 && second_check3 <= 1 && second_check4 >= 0){
                addBoundaryParticles(p.x+projection*line.nx, p.y+projection*line.ny, line, p, boundary_pairs);
            }
            else if((distance = sqrt((line.x1-p.x)*(line.x1-p.x) + (line.y1-p.y)*(line.y1-p.y))) < SMOOTH && ((line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny) > 0){
                addCornerBoundaryParticle(line.x1, line.y1, distance, p, boundary_pairs);
            }
            else if((distance = sqrt((line.x2-p.x)*(line.x2-p.x) + (line.y2-p.y)*(line.y2-p.y))) < SMOOTH && ((line.x2-p.x)*line.nx+(line.y2-p.y)*line.ny) > 0){
                addCornerBoundaryParticle(line.x2, line.y2, distance, p, boundary_pairs);
            }
        }
    }

    // Correct density of particles near boundaries
    for(int i=0; i<boundary_pairs.size(); i++){
        BoundaryPair &p = boundary_pairs[i];
        p.q = (float)(2*exp( -p.dist*p.dist / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
        p.q2 = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -p.dist*p.dist / (SMOOTH*SMOOTH/4)));
        p.a->dens += (p.boundary_mass)*p.q2;
    }
    
    // Calculate the pressure of each particle based on their density
    for (int i = 0; i < NUMPOINTS; i++) {
        Particle &p = particles[i];
        p.press = STIFF * (p.dens - REST);
    }

    // Calculate the acceleration of each particle caused by the surrounding particles
    // https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1
    for (int i = 0; i < pairs.size(); i++) {
        Pair &p = pairs[i];
        float press = M_P*(p.a->press/(p.a->dens*p.a->dens) + p.b->press/(p.b->dens*p.b->dens));
        float displace = (press * p.q) * (INTERVAL_MILI/1000.0);
        float abx = (p.a->x - p.b->x);
        float aby = (p.a->y - p.b->y);
        p.a->velx += displace * abx;
        p.a->vely += displace * aby;
        p.b->velx -= displace * abx;
        p.b->vely -= displace * aby;
    }
    
    // Calculate the acceleration of each particle caused by the surrounding boundaries
    // https://cg.informatik.uni-freiburg.de/publications/2017_VRIPHYS_MLSBoundaries.pdf
    for (int i = 0; i < boundary_pairs.size(); i++) {
        BoundaryPair &p = boundary_pairs[i];
        float press = (p.boundary_mass)*(p.a->press/(p.a->dens*p.a->dens));
        float displace = (press * p.q) * (INTERVAL_MILI/1000.0);
        float abx = (p.a->x - p.boundary_x);
        float aby = (p.a->y - p.boundary_y);
        p.a->velx += displace * abx;
        p.a->vely += displace * aby;
    }
    
    // Update the position of particles based on Euler's equation for an ideal fluid
    for (int i = 0; i < NUMPOINTS; i++) {
        Particle &p = particles[i];
        p.x += p.velx * (INTERVAL_MILI/1000.0);
        p.y += p.vely * (INTERVAL_MILI/1000.0);
    }
}

void physicsBackgroundThread(std::atomic<bool> &exit, std::atomic<bool> &updateRequired, std::atomic<int> &drawingIndex, Boundary* boundaries, Particle* particles, HWND m_hwnd){
    float velocity = 0;
    while(!exit.load()){
        bool expected = true;
        if(updateRequired.compare_exchange_weak(expected, false)){
            // If an update is necessary, update the particles UPDATES_PER_RENDER times and then redraw the particles
            for(int i=0; i<UPDATES_PER_RENDER; i++) updateParticles(drawingIndex, boundaries, particles);
            drawingIndex.store(0);

            // Redraw the particles
            InvalidateRect(m_hwnd, NULL, FALSE);
        }
    }
}