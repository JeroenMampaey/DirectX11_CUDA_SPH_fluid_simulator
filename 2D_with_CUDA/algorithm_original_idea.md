# Rtx 3060 Laptop GPU details:

- Ampere architecture
- 30 SMs
- 8.6 compute capability:
    - maximally 48 warps per SM
    - maximally 16 threadblocks per SM
    - maximally 100 KB shared memory per SM
    - threadblock recources usually seem to take up 1 KB of shared memory aswell

# First approach

## Idea

We use the exact algorithm from the [2D_no_CUDA](../2D_no_CUDA/) code but convert it to CUDA thus trying to parallelize it. Every particle uses a single thread and boundaries and pumps will be stored in shared memory and neighbour particles per thread will be stored in shared memory aswell. Because shared memory is limited in size, we will only store indexes to other particles so that instead of having to store two floats (resulting in 8 bytes) we need only to store a single short (2 bytes, since we will expect to use less than 65535 particles). 

## Constraints

Each threadblock will hold all the boundaries and pumps in shared memory requiring 8 bytes per boundary (x1,y1,x2,y2) and 12 bytes per 
pump (x1,y1,x2,y2,velx,vely), assuming at most 100 boundaries and 15 pumps this results in ~1 KB per threadblock. Each thread on it's own will allocate indexes to nearby boundaries in shared memory (1 byte per index) and indexes to neighbouring particles (2 bytes per index). Experimentally a particle usually has less than 35 neighbours and normally these particles are the one's that are not near to boundaries (particles near to boundaries always have less neighbours) thus if each thread could use 95 bytes and split these bytes between neighbour indexes and boundary indexes dynamically, this should be more than enough. Thus here I calculate number of threadblocks and warps per threadblock 
(ignoring register constraints since that's difficult to factor in):

Assuming k threadblocks per SM and m warps per threadblock the shared memory constraint gives us the following:

(1)
100000 >= k * (1000 + 1000) + k * m * 32 * 95
       = k * 2000 + k * m * 3040
       = k * (2000 + m * 3040)

Also if we require that atleast N particles can be used we need atleast this many threads thus:

(2)
30 * k * m * 32 >= N
<->
k*m >= N/960

But we also have because of the warp constraint that:

(3)
N <= 30 * 48 * 32
  = 46080

And then we also have the obvious constraints:

(4)
k * m <= 48

(5)
k <= 16

And now knowing this we want to maximize N while keeping reasonable values for k and m. To achieve this I'll try to simply plug some 
values:

- k=8:
  Then (1) gives that 
  
  3.45 >= m

  Thus to maximize N we find from (2) that we need to maximize m and thus m=3 achieving an occupancy of 24 warps per SM a.k.a 50% 
  and N=23040, even though the occupancy is suboptimal we still have a reasonable amount of threadblocks and a good amount of particles. 
  The number of bytes per thread here is instead of 95 bytes, 109 bytes which is pretty good.

- k=4:
  Then (1) gives that

  7.56 >= m

  Thus to maximize N we find from (2) that we need to maximize m and thus m=7 achieving an occupancy of 28 warps per SM a.k.a 58% 
  and N=26880, thus achieving a very similar occupancy and number of particles while drastically reducing the number of threadblocks 
  which is less optimal. The number of bytes per thread here is instead of 95 bytes, 102 bytes which is good but once again not as good 
  as when using k=8.

- k=2:
  Then (1) gives that

  15.78 >= m

  Thus to maximize N we find from (2) that we need to maximize m and thus m=15 achieving an occupancy of 30 warps per SM a.k.a 62.5% 
  and N=28800. The number of bytes per thread here is instead of 95 bytes, 100 bytes.

- k=1
  Then (1) gives that

  32.23 >= m

  Thus to maximize N we find from (2) that we need to maximize m and thus m=32 achieving an occupancy of 32 warps per SM a.k.a 66% 
  and N=30720. The number of bytes per thread here is exactly 95 bytes.

## Improvements

Thus k=8 and m=3 seems the most attractive option where the only real downside is the low amount of occupancy. Ideally we would maybe 
like to divide tasks even more to increase the occupancy without really increasing shared memory usage but this is difficult, at the 
moment every thread represents a particle so how could be possibly divide work further? Lets explore some ideas:

- 2 threads per particle, one handling real neighbours and one handling virtual neighbours: this won't work because once again to calculate virtual neighbours the thread already needs to know all the real neighbours

- 2 threads per particle, one handling the left side of the particle and one the right: this approach has the issue that both threads will arrive at incorrect density values thus these threads don't know the actual pressure density ratio. There are two solutions:
  - instead of storing the density pressure ratio, we could store only density and then both threads will simply need to add 
  their densities together when writing to global memory thus requiring some synchronization when finally writing to memory.
  - we could get threads performing on the left side and on the right side of a particle in the same warp and then perform a 
  simple warp shuffle to pass densities between threads before writing to global memory.

- 2 threads per particle, one handling particles with index lower than N/2 and the other handling particles with an index higher than N/2: This approach seems more logical than the previous approach but there are issues to be considered. First of, in the previous example we could simply halve shared memory usage per thread since particles using all the shared memory for storing there neighbours are usually particles fully surrounded by other particles thus having an equal number of particles to the left as to the right but when splitting the particles based on index, this could work out badly. To solve this we would need to use the same buffer for both threads so that they can dynamically take up a part of this buffer but this would require synchronization of some sort. Secondly this approach will result in more warp divergence (certain threads in warps will access different memory at the same time, the entire syncrhonization of sharing a buffer between 2 threads etc.) so it remains to be seen whether we actually get a decent speedup or not.

- 2 thread per particle, splitting neighbouring particles in two groups and each handling a single group: This approach works similar to approach number 2 but when looking for neighbours we simply need to give all oddly found neighbours to thread 1 and all evenly found neighbours to thread 2.

All these approaches thus can increase occupancy but it remains to be seen whether these actually improve performance in practice.