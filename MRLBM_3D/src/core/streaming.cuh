#pragma once

#include "../config.h"
#include RECONSTRUCT

__device__ inline void save_pop(real *s_pop, const real *pop)
{
    // save populations in shared memory
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;

#pragma unroll
    for (int q = 0; q < Q; q++)
    {
        s_pop[idxPopBlock(tx, ty, tz, q)] = pop[q];
    }
}

__device__ inline void streaming(const real *s_pop, real *pop)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;

#pragma unroll
    for (int q = 0; q < Q; q++)
    {
        int xs = (tx - d_cx[q] + BLOCK_THREAD_X) % BLOCK_THREAD_X;
        int ys = (ty - d_cy[q] + BLOCK_THREAD_Y) % BLOCK_THREAD_Y;
        int zs = (tz - d_cz[q] + BLOCK_THREAD_Z) % BLOCK_THREAD_Z;

        pop[q] = s_pop[idxPopBlock(xs, ys, zs, q)];
    }
}
