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

__device__ __forceinline__ size_t idxPopX(
    const unsigned int ty,  // Local y-thread index
    const unsigned int tz,  // Local z-thread index
    const unsigned int pop, // Which population index (0-8 for D3Q27 faces)
    const unsigned int bx,  // Block x-index
    const unsigned int by,  // Block y-index
    const unsigned int bz)  // Block z-index
{
    const size_t block_id = bx + (size_t)GRID_BLOCK_X * (by + (size_t)GRID_BLOCK_Y * bz);
    const size_t pop_id = pop + (size_t)QF * block_id;
    return (size_t)ty + (size_t)BLOCK_THREAD_Y * (tz + (size_t)BLOCK_THREAD_Z * pop_id);
}

__device__ __forceinline__ size_t idxPopY(
    const unsigned int tx,  // Local x-thread index
    const unsigned int tz,  // Local z-thread index
    const unsigned int pop, // Which population index
    const unsigned int bx,  // Block x-index
    const unsigned int by,  // Block y-index
    const unsigned int bz)  // Block z-index
{
    const size_t block_id = (size_t)bx + (size_t)GRID_BLOCK_X * ((size_t)by + (size_t)GRID_BLOCK_Y * bz);
    const size_t pop_id = (size_t)pop + (size_t)QF * block_id;
    return (size_t)tx + (size_t)BLOCK_THREAD_X * ((size_t)tz + (size_t)BLOCK_THREAD_Z * pop_id);
}

__device__ __forceinline__ size_t idxPopZ(
    const unsigned int tx,  // Local x-thread index
    const unsigned int ty,  // Local y-thread index
    const unsigned int pop, // Which population index
    const unsigned int bx,  // Block x-index
    const unsigned int by,  // Block y-index
    const unsigned int bz)  // Block z-index
{
    const size_t block_id = (size_t)bx + (size_t)GRID_BLOCK_X * ((size_t)by + (size_t)GRID_BLOCK_Y * bz);
    const size_t pop_id = (size_t)pop + (size_t)QF * block_id;
    return (size_t)tx + (size_t)BLOCK_THREAD_X * ((size_t)ty + (size_t)BLOCK_THREAD_Y * pop_id);
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