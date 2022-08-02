#include "physics.h"
#include "neighbor.h"
#include <vector>

//TODO:
// -remove sqrt's
// -move some calculations out of loop

// Helper function for the addGhostParticles function
void addGhostParticle(Particle &p, Boundary &line, Particle* neighbor_particle, float neighbor_x, float neighbor_y, int index){
    float projection = (line.x1 - neighbor_x)*line.nx + (line.y1 - neighbor_y)*line.ny;
    float virtual_x = neighbor_x + 2*projection*line.nx;
    float virtual_y = neighbor_y + 2*projection*line.ny;
    float first_check = ((p.x-line.x1)*line.nx+(p.y-line.y1)*line.ny)*((virtual_x-line.x1)*line.nx+(virtual_y-line.y1)*line.ny);
    if(first_check > 0) return;
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
    p.virtual_neighbors[index].neighbors.push_back(Neighbor(neighbor_particle, dist, virtual_x, virtual_y, q, q2));
}

// Add ghost particles for particle p over boundary line
void addGhostParticles(Particle &p, Boundary &line){
    int index = p.virtual_neighbors.size();
    p.virtual_neighbors.push_back(VirtualNeigbors(line.nx, line.ny));
    for(Neighbor neighbor : p.particle_neighbors){
        addGhostParticle(p, line, neighbor.p, neighbor.x, neighbor.y, index);
    }
    addGhostParticle(p, line, &p, p.x, p.y, index);
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
        p.x = crossing_x - 3.0*line.nx;
        p.y = crossing_y - 3.0*line.ny;
        p.velx = 0;
        p.vely = 0;
        return;
    }
}

void updateParticles(std::atomic<int> &drawingIndex, Boundary* boundaries, int numboundaries, Particle* particles, int numpoints){
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
            // If another particle is close enough to be a neighbor, first check if there is no boundary between the two particles
            bool crosses_boundary = false;
            for(int i=0; i<numboundaries; i++){
                Boundary &line = boundaries[i];
                float first_check = ((p.x-line.x1)*line.nx+(p.y-line.y1)*line.ny)*((p2.x-line.x1)*line.nx+(p2.y-line.y1)*line.ny);
                if(first_check > 0) continue;
                float second_check1 = (line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny;
                float second_check2 = (p2.x-p.x)*line.nx+(p2.y-p.y)*line.ny;
                float crossing_x = p.x;
                float crossing_y = p.y;
                if(second_check2 > 0.0){
                    crossing_x = p.x+(p2.x-p.x)*second_check1/second_check2;
                    crossing_y = p.y+(p2.y-p.y)*second_check1/second_check2;
                }
                float second_check3 = sqrt((crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1))/line.length;
                float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
                if(second_check3>1.0 || second_check4<0.0) continue;
                crosses_boundary = true;
            }
            if(crosses_boundary) continue;

            // Particle p2 is a neighbor of p
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
        // Look for boundaries near this particle
        for(int j=0; j<numboundaries; j++){
            Boundary &line = boundaries[j];
            float projection = (line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny;
            if(projection >= 0){
                float crossing_x = p.x + projection*line.nx;
                float crossing_y = p.y + projection*line.ny;
                float second_check3 = sqrt((crossing_x-line.x1)*(crossing_x-line.x1)+(crossing_y-line.y1)*(crossing_y-line.y1))/line.length;
                float second_check4 = (crossing_x-line.x1)*line.px+(crossing_y-line.y1)*line.py;
                bool particle_is_near_to_line = projection <= SMOOTH && second_check3 <= 1 && second_check4 >= 0;
                bool particle_is_near_to_endpoint1 = sqrt((line.x1-p.x)*(line.x1-p.x) + (line.y1-p.y)*(line.y1-p.y)) < SMOOTH && ((line.x1-p.x)*line.nx+(line.y1-p.y)*line.ny) > 0;
                bool particle_is_near_to_endpoint2 = sqrt((line.x2-p.x)*(line.x2-p.x) + (line.y2-p.y)*(line.y2-p.y)) < SMOOTH && ((line.x2-p.x)*line.nx+(line.y2-p.y)*line.ny) > 0;
                if(particle_is_near_to_line || particle_is_near_to_endpoint1 || particle_is_near_to_endpoint2){
                    // particle is near enough to the boundary to require ghost particlesd
                    addGhostParticles(p, line);
                }
            }
        }
    }
    
    // Calculate the pressure of each particle based on their density
    for (int i = 0; i < numpoints; i++) {
        Particle &p = particles[i];
        p.press = STIFF * (p.dens - REST);
    }

    // Calculate all the forces caused by the Euler equation
    for(int i = 0; i<numpoints; i++){
        Particle &p = particles[i];
        // Calculate velocity changes caused by other regular particles
        for(Neighbor neighbor : p.particle_neighbors){
            float press = M_P*(p.press/(p.dens*p.dens) + neighbor.p->press/(neighbor.p->dens*neighbor.p->dens));
            float displace = (press * neighbor.q) * (INTERVAL_MILI/1000.0);
            float abx = (p.x - neighbor.x);
            float aby = (p.y - neighbor.y);
            p.velx += displace * abx;
            p.vely += displace * aby;
        }

        // Calculate velocity changes caused by boundaries
        for(VirtualNeigbors bn : p.virtual_neighbors){
            float velx_change = 0.0;
            float vely_change = 0.0;
            for(Neighbor neighbor : bn.neighbors){
                float press = M_P*(p.press/(p.dens*p.dens) + neighbor.p->press/(neighbor.p->dens*neighbor.p->dens));
                float displace = (press * neighbor.q) * (INTERVAL_MILI/1000.0);
                float abx = (p.x - neighbor.x);
                float aby = (p.y - neighbor.y);
                velx_change += displace * abx;
                vely_change += displace * aby;
            }
            // Only allow a boundary to cause a velocity change if the particle is repulsed by it 
            // since boundaries should not be able to attract particles
            if(velx_change*bn.boundary_nx + vely_change*bn.boundary_ny <= 0.0){
                p.velx += velx_change;
                p.vely += vely_change;
            }
        }
    }
    
    // Update the position of particles based on Euler's equation for an ideal fluid
    for (int i = 0; i < numpoints; i++) {
        Particle &p = particles[i];
        float strength = sqrt(p.velx*p.velx + p.vely*p.vely);
        // Put a velocity limit on the particles too allow the system to work still somewhat normally 
        // if some unforeseen behaviour would occur
        if(strength > VEL_LIMIT){
            p.velx *= (VEL_LIMIT/strength);
            p.vely *= (VEL_LIMIT/strength);
        }
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