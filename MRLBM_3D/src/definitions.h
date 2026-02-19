#ifndef DEFINITIONS_H
#define DEFINITIIONS_H

#include "config.h"
// #include LATTICE_PROPERTIES
// #include CASE_CONSTANTS

constexpr size_t BYTES_PER_GB = (1 << 30);
constexpr size_t BYTES_PER_MB = (1 << 20);
constexpr size_t BYTES_PER_KB = (1 << 10);

constexpr size_t MAX_SHARED_MEM_BYTES = 48 * BYTES_PER_KB; // 48 kb shared memory
constexpr size_t SHARED_MEM_PER_THREAD = (Q) * sizeof(real);

constexpr dim3 OPTIMAL_BLOCK = findOptimalBlockDim3D(MAX_SHARED_MEM_BYTES, SHARED_MEM_PER_THREAD); // optimal block size that fits into 42kb of shared memory

// constexpr size_t BLOCK_THREAD_X = OPTIMAL_BLOCK.x; // Number of threads in x direction
// constexpr size_t BLOCK_THREAD_Y = OPTIMAL_BLOCK.y; //// Number of threads in y direction
// constexpr size_t BLOCK_THREAD_Z = OPTIMAL_BLOCK.z; //// Number of threads in z direction

constexpr size_t BLOCK_THREAD_X = 8; // Number of threads in x direction
constexpr size_t BLOCK_THREAD_Y = 8; //// Number of threads in y direction
constexpr size_t BLOCK_THREAD_Z = 4; //// Number of threads in z direction

constexpr size_t GRID_BLOCK_X = (NX + BLOCK_THREAD_X - 1) / BLOCK_THREAD_X; // Number of blocks in x direction
constexpr size_t GRID_BLOCK_Y = (NY + BLOCK_THREAD_Y - 1) / BLOCK_THREAD_Y; // Number of blocks in y direction
constexpr size_t GRID_BLOCK_Z = (NZ + BLOCK_THREAD_Z - 1) / BLOCK_THREAD_Z; // Number of blocks in z direction

// constexpr size_t GRID_BLOCK_X = NX  / BLOCK_THREAD_X; // Number of blocks in x direction
// constexpr size_t GRID_BLOCK_Y = NY  / BLOCK_THREAD_Y; // Number of blocks in y direction
// constexpr size_t GRID_BLOCK_Z = NZ  / BLOCK_THREAD_Z; // Number of blocks in z direction

constexpr size_t THREADS_PER_BLOCK = BLOCK_THREAD_X * BLOCK_THREAD_Y * BLOCK_THREAD_Z; // total number of threads in a single block
constexpr size_t NUMBER_OF_BLOCKS = GRID_BLOCK_X * GRID_BLOCK_Y * GRID_BLOCK_Z;        // total number of blocks in whole domain
constexpr size_t TOTAL_NUMBER_OF_THREADS = NUMBER_OF_BLOCKS * THREADS_PER_BLOCK;       // total number of threads in whole domain

constexpr dim3 block(BLOCK_THREAD_X, BLOCK_THREAD_Y, BLOCK_THREAD_Z); // Number of Block in x, y, z in the grid
constexpr dim3 grid(GRID_BLOCK_X, GRID_BLOCK_Y, GRID_BLOCK_Z);        // Number of threads in x, y, z in a block

constexpr size_t BLOCK_FACE_XY = BLOCK_THREAD_X * BLOCK_THREAD_Y;
constexpr size_t BLOCK_FACE_XZ = BLOCK_THREAD_X * BLOCK_THREAD_Z;
constexpr size_t BLOCK_FACE_YZ = BLOCK_THREAD_Y * BLOCK_THREAD_Z;
constexpr size_t BLOCK_HALO_SIZE = 2 * (BLOCK_FACE_XY + BLOCK_FACE_XZ + BLOCK_FACE_YZ);

constexpr size_t NUM_LBM_NODES = TOTAL_NUMBER_OF_THREADS;
constexpr size_t NUM_HALO_FACE_XY = BLOCK_FACE_XY * NUMBER_OF_BLOCKS;
constexpr size_t NUM_HALO_FACE_XZ = BLOCK_FACE_XZ * NUMBER_OF_BLOCKS;
constexpr size_t NUM_HALO_FACE_YZ = BLOCK_FACE_YZ * NUMBER_OF_BLOCKS;
constexpr size_t TOTAL_HALO_NODES = BLOCK_HALO_SIZE * NUMBER_OF_BLOCKS;

constexpr size_t USED_SHARED_MEMORY = SHARED_MEM_PER_THREAD * THREADS_PER_BLOCK/ BYTES_PER_KB;
constexpr size_t USED_GLOBAL_MEMORY = NUM_LBM_NODES * ((NUMBER_OF_MOMENTS * sizeof(real)) + sizeof(nodeType_t)) / BYTES_PER_MB;
constexpr size_t HALO_GLOBAL_MEMORY = TOTAL_HALO_NODES * QF * sizeof(real) / BYTES_PER_MB;

#endif // DEFINITIONS_H