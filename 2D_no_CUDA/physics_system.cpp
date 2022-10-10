#include "physics_system.h"

#define GRAVITY 9.8f
#define PIXEL_PER_METER 100.0f
#define SMOOTH (3.0f*RADIUS)
#define REST 1
#define STIFF 50000.0
#define M_P (REST*RADIUS*RADIUS*PI)
#define VEL_LIMIT 8.0

PhysicsSystem::PhysicsSystem(GraphicsEngine& gfx){
    float refreshRate;
    if(RATE_IS_INVALID(refreshRate = gfx.getRefreshRate())){
        throw std::exception("Refreshrate could not easily be found programmatically.");
    }
    dt = 1.0f/(UPDATES_PER_RENDER*refreshRate);
}

void PhysicsSystem::update(EntityManager& manager) const noexcept{
    for(int i=0; i<UPDATES_PER_RENDER; i++){
        performNonFluidRelatedPhysics(manager);
        updateDensityFieldCausedByNeighbours(manager);
        updateDensityFieldCausedByGhostParticles(manager);
        updatePressureField(manager);
        applyEulersEquation(manager);
    }
}

inline void PhysicsSystem::performNonFluidRelatedPhysics(EntityManager& manager) const noexcept{
    for(Particle& p : manager.getParticles()){
        p.vel.x = (p.pos.x - p.oldPos.x) / (PIXEL_PER_METER*dt);
        p.vel.y = (p.pos.y - p.oldPos.y) / (PIXEL_PER_METER*dt);

        // Update velocities based on whether the particle is in a pump or not
        for(const Pump& pump : manager.getPumps()){
            if(p.pos.x >= pump.leftBottom.x && p.pos.x <= pump.rightTop.x && p.pos.y >= pump.leftBottom.y && p.pos.y <= pump.rightTop.y){
                p.vel.x = pump.vel.x;
                p.vel.y = pump.vel.y;
                break;
            }
        }
        
        // Update positional change of particles caused by gravity
        p.vel.y -= GRAVITY*dt;

        // Update particle positions
        p.pos.x += PIXEL_PER_METER*p.vel.x*dt;
        p.pos.y += PIXEL_PER_METER*p.vel.y*dt;

        // Update positional change of particles caused by boundaries (make sure particles cannot pass boundaries)
        for(const Boundary& line : manager.getBoundaries()){
            float first_check = ((p.pos.x-line.point1.x)*line.normal.x+(p.pos.y-line.point1.y)*line.normal.y)*((p.oldPos.x-line.point1.x)*line.normal.x+(p.oldPos.y-line.point1.y)*line.normal.y);
            // Check if pos and oldPos lie on the same side of the boundary
            if(first_check > 0.0f) continue;
            float second_check1 = (line.point1.x-p.oldPos.x)*line.normal.x+(line.point1.y-p.oldPos.y)*line.normal.y;
            // Check if new position of particle lies outside the boundary (normal of boundary points to the outside)
            if(second_check1 < 0) continue;
            float second_check2 = (p.pos.x-p.oldPos.x)*line.normal.x+(p.pos.y-p.oldPos.y)*line.normal.y;
            // Calculate where the particle crossed the boundary
            float crossing_x = p.oldPos.x;
            float crossing_y = p.oldPos.y;
            if(second_check2 > 0.0){
                crossing_x += (p.pos.x-p.oldPos.x)*second_check1/second_check2;
                crossing_y += (p.pos.y-p.oldPos.y)*second_check1/second_check2;
            }
            float second_check3 = (crossing_x-line.point1.x)*(crossing_x-line.point1.x)+(crossing_y-line.point1.y)*(crossing_y-line.point1.y);
            float second_check4 = (crossing_x-line.point1.x)*line.direction.x+(crossing_y-line.point1.y)*line.direction.y;
            // Check if the crossing point is actually in the boundary (and not next to the boundary)
            if(second_check3>line.lengthSquared || second_check4<0.0) continue;
            // Put particle back above the crossing point
            p.pos.x = crossing_x - RADIUS*line.normal.x;
            p.pos.y = crossing_y - RADIUS*line.normal.y;
            p.vel.x = 0.0;
            p.vel.y = 0.0;
            break;
        }

        // Store the old particle positions
        p.oldPos.x = p.pos.x - PIXEL_PER_METER*p.vel.x*dt;
        p.oldPos.y = p.pos.y - PIXEL_PER_METER*p.vel.y*dt;
        
        const float kernel = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( 0 / (SMOOTH*SMOOTH/4)));
        p.dens = M_P*kernel;
        p.vel.x = 0.0f;
        p.vel.y = 0.0f;
        p.neighbours.clear();
        p.virtualNeighbours.clear();
    }
}

inline void PhysicsSystem::updateDensityFieldCausedByNeighbours(EntityManager& manager) const noexcept{
    for(int i=0; i<manager.getParticles().size(); i++){
        Particle& p = manager.getParticles()[i];
        // Find particles that are near to p and update densities
        for (int j = 0; j < i; j++) {
            Particle &p2 = manager.getParticles()[j];
            float dist_squared = (p.pos.x-p2.pos.x)*(p.pos.x-p2.pos.x)+(p.pos.y-p2.pos.y)*(p.pos.y-p2.pos.y);
            if (dist_squared < SMOOTH*SMOOTH) {
                // If another particle is close enough to be a neighbor, first check if there is no boundary between the two particles
                int e=0;
                for(; e<manager.getBoundaries().size(); e++){
                    const Boundary &line = manager.getBoundaries()[e];
                    float first_check = ((p.pos.x-line.point1.x)*line.normal.x+(p.pos.y-line.point1.y)*line.normal.y)*((p2.pos.x-line.point1.x)*line.normal.x+(p2.pos.y-line.point1.y)*line.normal.y);
                    // Check if p and p2 lie on the same side of the boundary
                    if(first_check > 0) continue;
                    float second_check1 = (line.point1.x-p.pos.x)*line.normal.x+(line.point1.y-p.pos.y)*line.normal.y;
                    float second_check2 = (p2.pos.x-p.pos.x)*line.normal.x+(p2.pos.y-p.pos.y)*line.normal.y;
                    // Calculate where the path between p and p2 crosses the boundary
                    float crossing_x = p.pos.x;
                    float crossing_y = p.pos.y;
                    if(second_check2 > 0.0){
                        crossing_x += (p2.pos.x-p.pos.x)*second_check1/second_check2;
                        crossing_y += (p2.pos.y-p.pos.y)*second_check1/second_check2;
                    }
                    float second_check3 = (crossing_x-line.point1.x)*(crossing_x-line.point1.x)+(crossing_y-line.point1.y)*(crossing_y-line.point1.y);
                    float second_check4 = (crossing_x-line.point1.x)*line.direction.x+(crossing_y-line.point1.y)*line.direction.y;
                    // Check if the crossing point is actually in the boundary (and not next to the boundary)
                    if(second_check3>line.lengthSquared || second_check4<0.0) continue;
                    break;
                }
                // If path between p and p2 crosses a boundary, look at another p2
                if(e<manager.getBoundaries().size()) continue;
                
                // Particle p2 is a neighbor of p, add eachother to eachothers neighbour lists and calculate density changes
                float kernel = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist_squared / (SMOOTH*SMOOTH/4)));
                float gradKernel = (float)(2*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
                p.dens += M_P*kernel;
                p2.dens += M_P*kernel;
                p.neighbours.push_back(Neighbour(p2, kernel, gradKernel));
                p2.neighbours.push_back(Neighbour(p, kernel, gradKernel));
            }
        }
    }
}

inline void PhysicsSystem::updateDensityFieldCausedByGhostParticles(EntityManager& manager) const noexcept{
    for(Particle& p : manager.getParticles()){
        // Look for boundaries near this particle to be able to add necessary ghost particles
        for(const Boundary& line : manager.getBoundaries()){
            float projection = (line.point1.x-p.pos.x)*line.normal.x+(line.point1.y-p.pos.y)*line.normal.y;
            // Check if particle is on the inside of this boundary (normal of boundary points to the outside)
            if(projection >= 0){
                // Calculate the point on the boundary which the particle is hovering above
                float crossing_x = p.pos.x + projection*line.normal.x;
                float crossing_y = p.pos.y + projection*line.normal.y;
                float second_check3 = (crossing_x-line.point1.x)*(crossing_x-line.point1.x)+(crossing_y-line.point1.y)*(crossing_y-line.point1.y);
                float second_check4 = (crossing_x-line.point1.x)*line.direction.x+(crossing_y-line.point1.y)*line.direction.y;
                if(projection <= SMOOTH && second_check3 <= line.lengthSquared && second_check4 >= 0){
                    // Particle is hovering above this boundary and distance from boundary is less than SMOOTH
                }
                else if(((line.point1.x-p.pos.x)*(line.point1.x-p.pos.x) + (line.point1.y-p.pos.y)*(line.point1.y-p.pos.y)) < SMOOTH*SMOOTH && ((line.point1.x-p.pos.x)*line.normal.x+(line.point1.y-p.pos.y)*line.normal.y) > 0){
                    // Particle is close enough to one endpoint of the boundary
                }
                else if(((line.point2.x-p.pos.x)*(line.point2.x-p.pos.x) + (line.point2.y-p.pos.y)*(line.point2.y-p.pos.y)) < SMOOTH*SMOOTH && ((line.point2.x-p.pos.x)*line.normal.x+(line.point2.y-p.pos.y)*line.normal.y) > 0){
                    // Particle is close enough to another endpoint of the boundary
                }
                else{
                    continue;
                }

                // Use the neighbours of the particle to calculate ghost particle positions
                p.virtualNeighbours.push_back({{}, line});
                for(const Neighbour& neighbour : p.neighbours){
                    addGhostParticleHelper(p, line, neighbour.p);
                }
                addGhostParticleHelper(p, line, p);
            }
        }
    }
}

inline void PhysicsSystem::updatePressureField(EntityManager& manager) const noexcept{
    // Calculate the pressure of each particle based on their density
    for(Particle& p : manager.getParticles()){
        if(p.dens>0.0){
            p.pressure_density_ratio = STIFF*(p.dens-REST)/(p.dens*p.dens);
        }
    }
}

inline void PhysicsSystem::applyEulersEquation(EntityManager& manager) const noexcept{
    for(Particle& p : manager.getParticles()){
        // Calculate velocity changes caused by other regular particles
        for(const Neighbour& neighbour : p.neighbours){
            float press = M_P*(p.pressure_density_ratio + neighbour.p.pressure_density_ratio);
            float displace = (press * neighbour.gradKernel) * dt;
            float abx = (p.pos.x - neighbour.p.pos.x);
            float aby = (p.pos.y - neighbour.p.pos.y);
            p.vel.x += displace * abx;
            p.vel.y += displace * aby;
        }

        // Calculate velocity changes caused by boundaries
        for(const std::pair<std::vector<VirtualNeighbour>, const Boundary&> pair : p.virtualNeighbours){
            float velx_change = 0.0f;
            float vely_change = 0.0f;
            for(const VirtualNeighbour neighbour : pair.first){
                float press = M_P*(p.pressure_density_ratio + neighbour.p.pressure_density_ratio);
                float displace = (press * neighbour.gradKernel) * dt;
                float abx = (p.pos.x - neighbour.virtualX);
                float aby = (p.pos.y - neighbour.virtualY);
                velx_change += displace * abx;
                vely_change += displace * aby;
            }

            // Only allow a boundary to cause a velocity change if the particle is repulsed by it 
            // since boundaries should not be able to attract particles
            if(velx_change*pair.second.normal.x + vely_change*pair.second.normal.y <= 0.0){
                p.vel.x += velx_change;
                p.vel.y += vely_change;
            }
        }
    }

    // Update the position of particles based on Euler's equation for an ideal fluid
    for(Particle& p : manager.getParticles()){
        // Put a velocity limit on the particles too allow the system to work still somewhat normally 
        // if some unforeseen behaviour would occur
        float strength_squared = p.vel.x*p.vel.x + p.vel.y*p.vel.y;
        if(strength_squared > VEL_LIMIT*VEL_LIMIT){
            float normalization_constant = VEL_LIMIT/sqrt(strength_squared);
            p.vel.x *= normalization_constant;
            p.vel.y *= normalization_constant;
        }
        p.pos.x += PIXEL_PER_METER*p.vel.x*dt;
        p.pos.y += PIXEL_PER_METER*p.vel.y*dt;
    }
}

inline void PhysicsSystem::addGhostParticleHelper(Particle& p, const Boundary& line, const Particle& p2) const noexcept{
    float projection = (line.point1.x - p2.pos.x)*line.normal.x + (line.point1.y - p2.pos.y)*line.normal.y;
    // Calculate position of the ghost particle (by reflecting p2 over the boundary)
    float virtual_x = p2.pos.x + 2*projection*line.normal.x;
    float virtual_y = p2.pos.y + 2*projection*line.normal.y;
    float first_check = ((p.pos.x-line.point1.x)*line.normal.x+(p.pos.y-line.point1.y)*line.normal.y)*((virtual_x-line.point1.x)*line.normal.x+(virtual_y-line.point1.y)*line.normal.y);
    // Check if p and the ghost particle lie on the same side of the boundary
    if(first_check > 0) return;
    float second_check1 = (line.point1.x-p.pos.x)*line.normal.x+(line.point1.y-p.pos.y)*line.normal.y;
    float second_check2 = (virtual_x-p.pos.x)*line.normal.x+(virtual_y-p.pos.y)*line.normal.y;
    // Calculate where the path between p and the ghost particle crosses the boundary
    float crossing_x = p.pos.x;
    float crossing_y = p.pos.y;
    if(second_check2 > 0.0){
        crossing_x += (virtual_x-p.pos.x)*second_check1/second_check2;
        crossing_y += (virtual_y-p.pos.y)*second_check1/second_check2;
    }
    float second_check3 = (crossing_x-line.point1.x)*(crossing_x-line.point1.x)+(crossing_y-line.point1.y)*(crossing_y-line.point1.y);
    float second_check4 = (crossing_x-line.point1.x)*line.direction.x+(crossing_y-line.point1.y)*line.direction.y;
    // Check if the crossing point is actually in the boundary (and not next to the boundary)
    if(second_check3>line.lengthSquared || second_check4<0.0) return;
    float dist_squared = (virtual_x-p.pos.x)*(virtual_x-p.pos.x)+(virtual_y-p.pos.y)*(virtual_y-p.pos.y);
    // Check if distance to the virtual point in less than the smoothing kernel distance
    if(dist_squared > SMOOTH*SMOOTH) return;

    // The ghost particle is a neighbor of p, calculate density changes
    float kernel = (float)((1.0 / ((SMOOTH/2)*SQRT_PI))*(1.0 / ((SMOOTH/2)*SQRT_PI))*exp( -dist_squared / (SMOOTH*SMOOTH/4)));
    float gradKernel = (float)(2*exp( -dist_squared / (SMOOTH*SMOOTH/4)) / (SMOOTH*SMOOTH*SMOOTH*SMOOTH/16) / PI);
    p.dens += M_P*kernel;
    p.virtualNeighbours.back().first.push_back(VirtualNeighbour(p2, kernel, gradKernel, virtual_x, virtual_y));
}