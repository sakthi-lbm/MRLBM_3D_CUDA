#ifndef INDEX_H
#define INDEX_H

#include "config.h"

__host__ __device__ inline size_t IDX(unsigned int x, unsigned int y)
{
    return x + (y * NX);
}
__host__ __device__ __forceinline__ size_t IDX_BLOCK(const unsigned int tx,
                                                     const unsigned int ty,
                                                     const unsigned int tz,
                                                     const unsigned int bx,
                                                     const unsigned int by,
                                                     const unsigned int bz)
{
    const size_t threads_per_block =
        BLOCK_THREAD_X * BLOCK_THREAD_Y * BLOCK_THREAD_Z;

    const size_t block_index =
        bx + GRID_BLOCK_X * (by + GRID_BLOCK_Y * bz);

    const size_t thread_index =
        tx + BLOCK_THREAD_X * (ty + BLOCK_THREAD_Y * tz);

    return thread_index + threads_per_block * block_index;
}

__host__ __device__ __forceinline__
    size_t
    idxPopBlock(const unsigned int tx, const unsigned int ty, const unsigned int pop)
{
    const size_t popOffset = pop * THREADS_PER_BLOCK; // starting index of this population
    const size_t yOffset = ty * BLOCK_THREAD_X;       // offset for this row within population

    return tx + yOffset + popOffset; // final linear index
}

__device__ __forceinline__ size_t idxPopX(unsigned int ty, int pop, unsigned int bx, unsigned int by)
{
    const size_t block_id = bx + GRID_BLOCK_X * by; // stride to jump between blocks in the grid
    const size_t pop_id = pop + QF * block_id;      // stride to jump between populations in all blocks
    return ty + BLOCK_THREAD_Y * pop_id;            // stride to jump within a population (along local threads)
}

__device__ __forceinline__ size_t idxPopY(unsigned int tx, int pop, unsigned int bx, unsigned int by)
{
    const size_t block_id = bx + GRID_BLOCK_X * by; // stride to jump between blocks in the grid
    const size_t pop_id = pop + QF * block_id;      // stride to jump between populations in all blocks
    return tx + BLOCK_THREAD_X * pop_id;            // stride to jump within a population (along local threads)
}

__host__ __device__ __forceinline__ size_t idxBoundPop(size_t node, int q)
{
    return q + node * Q;
}

__host__ __device__ __forceinline__ void GlobalIndexToXY(const size_t global_index, unsigned int &x, unsigned int &y)
{
    const size_t threads_per_block = BLOCK_THREAD_X * BLOCK_THREAD_Y;

    const size_t block_index = global_index / threads_per_block;
    const size_t thread_index = global_index % threads_per_block;

    const size_t bx = block_index % GRID_BLOCK_X;
    const size_t by = block_index / GRID_BLOCK_X;

    const size_t tx = thread_index % BLOCK_THREAD_X;
    const size_t ty = thread_index / BLOCK_THREAD_X;

    x = bx * BLOCK_THREAD_X + tx;
    y = by * BLOCK_THREAD_Y + ty;
}

// for 3D

#endif // INDEX_H