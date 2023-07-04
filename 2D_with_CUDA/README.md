# How to use this program

This folder contains the files for a CUDA accelerated 2D SPH simulation program. The program provides a simple default simulation layout but will also first look if a custom simulation layout has been specified as "simulation2D.txt" in the [simulation_layout](../simulation_layout) folder, if a file with the correct name exists then this simulation layout will be used instead of the default.

The particles can be configured as having a custom size by changing the **RADIUS** macro in the [entity_manager.h](./entity_manager.h) file. A particle size less than 1.2f does however not seem to work anymore.

# How it works

Everytime particles are updated, a radix sort quickly sorts all particles based on the x-coordinate and then each threadblock is responsible for 64 particles from this sorted list. Sorting is done to allow particles to quickly look for possible neighbour particles. Of course, only sorting on x-coordinates can result in a worst case scenario where all particles are stacked on top of each other in the vertical direction for example, scenarios like this however are very unlikely to occur naturally since fluids tend to prefer staying level. On a high level, the loop looks like this:
- Update regular physics (gravity,...)
- Sort particles
- Update density, pressure,...

The sorting is done with [cub::DeviceRadixSort](https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html). At the moment though, there is still room for improvement:
- Sorting only needs to be done at the granularity of 64 particles, not down to the individual particle
- Resorting each time is inefficient since between two updates most particles barely move