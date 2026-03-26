#pragma once

#include "../config.h"

#include CASE_OUTPUTS

constexpr real PI = real(3.14159265358979323846);
constexpr real SQRT_2 = real(1.4142135623730950488);
constexpr real SQRT_3 = real(1.7320508075688772935);

constexpr size_t BYTES_PER_GB = (1 << 30);
constexpr size_t BYTES_PER_MB = (1 << 20);
constexpr size_t BYTES_PER_KB = (1 << 10);

constexpr size_t MAX_SHARED_MEM_BYTES = 48 * BYTES_PER_KB; // 48 kb shared memory
constexpr size_t SHARED_MEM_PER_THREAD = (Q) * sizeof(real);

constexpr size_t BLOCK_NODES = 256; // size of the blocks for boundary nodes

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

constexpr size_t USED_SHARED_MEMORY_BYTES = SHARED_MEM_PER_THREAD * THREADS_PER_BLOCK;
constexpr size_t USED_GLOBAL_MEMORY_BYTES = NUM_LBM_NODES * ((NUMBER_OF_MOMENTS * sizeof(real)) + sizeof(nodeType_t));
constexpr size_t HALO_GLOBAL_MEMORY_BYTES = TOTAL_HALO_NODES * QF * sizeof(real);

constexpr size_t USED_GLOBAL_MEMORY = USED_GLOBAL_MEMORY_BYTES / BYTES_PER_MB;
constexpr size_t HALO_GLOBAL_MEMORY = HALO_GLOBAL_MEMORY_BYTES / BYTES_PER_MB;
