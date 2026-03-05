#ifndef MLBM_H
#define MLBM_H

#include "solver/initializeLBM.cuh"
#include HALO_INTERFACE
#include STREAMING
#include EVAL_MOMENTS

__global__ void streaming_and_evaluate_Mom(const cylinderVar cylinder, nodeVar fMom, const int iter);
__global__ void collision_halo_update(const cylinderVar cylinder, nodeVar fMom, const int iter);

__device__ __forceinline__ void load_moments(const unsigned int sx,
                                             const unsigned int sy,
                                             const unsigned int sz,
                                             const size_t gidx,
                                             const nodeVar &dMom, moments &smem)
{
    // printf("Thread: sx=%u sy=%u sz=%u\n", sx, sy, sz);
    smem.rho[sz][sy][sx] = RHO_0 + dMom.rho[gidx];
    smem.ux[sz][sy][sx] = dMom.ux[gidx];
    smem.uy[sz][sy][sx] = dMom.uy[gidx];
    smem.uz[sz][sy][sx] = dMom.uz[gidx];

    smem.mxx[sz][sy][sx] = dMom.mxx[gidx];
    smem.myy[sz][sy][sx] = dMom.myy[gidx];
    smem.mzz[sz][sy][sx] = dMom.mzz[gidx];

    smem.mxy[sz][sy][sx] = dMom.mxy[gidx];
    smem.mxz[sz][sy][sx] = dMom.mxz[gidx];
    smem.myz[sz][sy][sx] = dMom.myz[gidx];
}

__device__ __forceinline__ void load_shared_moments(const nodeVar &dMom, moments &smem)
{
    const unsigned int sx = threadIdx.x + HALO;
    const unsigned int sy = threadIdx.y + HALO;
    const unsigned int sz = threadIdx.z + HALO;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;

    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int bz = blockIdx.z;

    // WEST
    int tx_west = tx - HALO;
    int bx_west = bx;
    if (tx_west < 0)
    {
        tx_west += BLOCK_THREAD_X;
        bx_west -= 1;
        if (bx_west < 0)
            bx_west = GRID_BLOCK_X - 1;
    }

    // EAST
    int tx_east = tx + HALO;
    int bx_east = bx;
    if (tx_east >= BLOCK_THREAD_X)
    {
        tx_east -= BLOCK_THREAD_X;
        bx_east += 1;
        if (bx_east >= GRID_BLOCK_X)
            bx_east = 0;
    }

    // SOUTH
    int ty_south = ty - HALO;
    int by_south = by;
    if (ty_south < 0)
    {
        ty_south += BLOCK_THREAD_Y;
        by_south -= 1;
        if (by_south < 0)
            by_south = GRID_BLOCK_Y - 1;
    }

    // NORTH
    int ty_north = ty + HALO;
    int by_north = by;
    if (ty_north >= BLOCK_THREAD_Y)
    {
        ty_north -= BLOCK_THREAD_Y;
        by_north += 1;
        if (by_north >= GRID_BLOCK_Y)
            by_north = 0;
    }

    // BACK
    int tz_back = tz - HALO;
    int bz_back = bz;
    if (tz_back < 0)
    {
        tz_back += BLOCK_THREAD_Z;
        bz_back -= 1;
        if (bz_back < 0)
            bz_back = GRID_BLOCK_Z - 1;
    }

    // FRONT
    int tz_front = tz + HALO;
    int bz_front = bz;
    if (tz_front >= BLOCK_THREAD_Z)
    {
        tz_front -= BLOCK_THREAD_Z;
        bz_front += 1;
        if (bz_front >= GRID_BLOCK_Z)
            bz_front = 0;
    }

    // ========================================= INTERIOR ==============================================
    load_moments(sx, sy, sz,
                 IDX_BLOCK(tx, ty, tz, bx, by, bz),
                 dMom, smem);

    __syncthreads();
    // ========================================= FACE HALOS ==============================================
    //------------------------- West -------------------------------
    if (tx == 0 && ty > 0 && ty < BLOCK_THREAD_Y - 1 && tz > 0 && tz < BLOCK_THREAD_Z - 1)
    {
        // printf("Thread: tx=%u ty=%u tz=%u|bx=%u by=%u bz=%u | tx_west=%d bx_west=%d\n",
        //        tx, ty, tz,bx,by,bz, tx_west, bx_west);
        load_moments(sx - HALO, sy, sz,
                     IDX_BLOCK(tx_west, ty, tz, bx_west, by, bz),
                     dMom, smem);
    }

    //------------------------- East -------------------------------
    if (tx == BLOCK_THREAD_X - 1 && ty > 0 && ty < BLOCK_THREAD_Y - 1 && tz > 0 && tz < BLOCK_THREAD_Z - 1)
    {
        load_moments(sx + HALO, sy, sz,
                     IDX_BLOCK(tx_east, ty, tz, bx_east, by, bz),
                     dMom, smem);
    }

    //------------------------- South -------------------------------
    if (ty == 0 && tx > 0 && tx < BLOCK_THREAD_X - 1 && tz > 0 && tz < BLOCK_THREAD_Z - 1)
    {
        load_moments(sx, sy - HALO, sz,
                     IDX_BLOCK(tx, ty_south, tz, bx, by_south, bz),
                     dMom, smem);
    }

    //------------------------- North -------------------------------
    if (ty == BLOCK_THREAD_Y - 1 && tx > 0 && tx < BLOCK_THREAD_X - 1 && tz > 0 && tz < BLOCK_THREAD_Z - 1)
    {
        load_moments(sx, sy + HALO, sz,
                     IDX_BLOCK(tx, ty_north, tz, bx, by_north, bz),
                     dMom, smem);
    }

    //------------------------- Back -------------------------------
    if (tz == 0 && tx > 0 && tx < BLOCK_THREAD_X - 1 && ty > 0 && ty < BLOCK_THREAD_Y - 1)
    {
        load_moments(sx, sy, sz - HALO,
                     IDX_BLOCK(tx, ty, tz_back, bx, by, bz_back),
                     dMom, smem);
    }

    //------------------------- Front-------------------------------
    if (tz == BLOCK_THREAD_Z - 1 && tx > 0 && tx < BLOCK_THREAD_X - 1 && ty > 0 && ty < BLOCK_THREAD_Y - 1)
    {
        load_moments(sx, sy, sz + HALO,
                     IDX_BLOCK(tx, ty, tz_front, bx, by, bz_front),
                     dMom, smem);
    }
    __syncthreads();
    // ========================================= EDGE HALOS ==============================================
    if (tx == 0 && ty == 0 && tz > 0 && tz < BLOCK_THREAD_Z - 1)
    {
        // WEST-SOUTH
        load_moments(sx - HALO, sy - HALO, sz,
                     IDX_BLOCK(tx_west, ty_south, tz, bx_west, by_south, bz),
                     dMom, smem);
    }
    if (tx == 0 && ty == BLOCK_THREAD_Y - 1 && tz > 0 && tz < BLOCK_THREAD_Z - 1)
    {
        // WEST-NORTH
        load_moments(sx - HALO, sy + HALO, sz,
                     IDX_BLOCK(tx_west, ty_north, tz, bx_west, by_north, bz),
                     dMom, smem);
    }
    if (tx == 0 && tz == 0 && ty > 0 && ty < BLOCK_THREAD_Y - 1)
    {
        // WEST-BACK
        load_moments(sx - HALO, sy, sz - HALO,
                     IDX_BLOCK(tx_west, ty, tz_back, bx_west, by, bz_back),
                     dMom, smem);
    }
    if (tx == 0 && tz == BLOCK_THREAD_Z - 1 && ty > 0 && ty < BLOCK_THREAD_Y - 1)
    {
        // WEST-FRONT
        load_moments(sx - HALO, sy, sz + HALO,
                     IDX_BLOCK(tx_west, ty, tz_front, bx_west, by, bz_front),
                     dMom, smem);
    }

    if (tx == BLOCK_THREAD_X - 1 && ty == 0 && tz > 0 && tz < BLOCK_THREAD_Z - 1)
    {
        // EAST-SOUTH
        load_moments(sx + HALO, sy - HALO, sz,
                     IDX_BLOCK(tx_east, ty_south, tz, bx_east, by_south, bz),
                     dMom, smem);
    }
    if (tx == BLOCK_THREAD_X - 1 && ty == BLOCK_THREAD_Y - 1 && tz > 0 && tz < BLOCK_THREAD_Z - 1)
    {
        // EAST-NORTH
        load_moments(sx + HALO, sy + HALO, sz,
                     IDX_BLOCK(tx_east, ty_north, tz, bx_east, by_north, bz),
                     dMom, smem);
    }
    if (tx == BLOCK_THREAD_X - 1 && tz == 0 && ty > 0 && ty < BLOCK_THREAD_Y - 1)
    {
        // EAST-BACK
        load_moments(sx + HALO, sy, sz - HALO,
                     IDX_BLOCK(tx_east, ty, tz_back, bx_east, by, bz_back),
                     dMom, smem);
    }
    if (tx == BLOCK_THREAD_X - 1 && tz == BLOCK_THREAD_Z - 1 && ty > 0 && ty < BLOCK_THREAD_Y - 1)
    {
        // EAST-FRONT
        load_moments(sx + HALO, sy, sz + HALO,
                     IDX_BLOCK(tx_east, ty, tz_front, bx_east, by, bz_front),
                     dMom, smem);
    }

    // --------------------------------------- YZ-Plane ----------------------------------------
    if (ty == 0 && tz == 0 && tx > 0 && tx < BLOCK_THREAD_X - 1)
    {
        // SOUTH-BACK
        load_moments(sx, sy - HALO, sz - HALO,
                     IDX_BLOCK(tx, ty_south, tz_back, bx, by_south, bz_back),
                     dMom, smem);
    }
    if (ty == 0 && tz == BLOCK_THREAD_Z - 1 && tx > 0 && tx < BLOCK_THREAD_X - 1)
    {
        // SOUTH-FRONT
        load_moments(sx, sy - HALO, sz + HALO,
                     IDX_BLOCK(tx, ty_south, tz_front, bx, by_south, bz_front),
                     dMom, smem);
    }
    if (ty == BLOCK_THREAD_Y - 1 && tz == 0 && tx > 0 && tx < BLOCK_THREAD_X - 1)
    {
        // NORTH-BACK
        load_moments(sx, sy + HALO, sz - HALO,
                     IDX_BLOCK(tx, ty_north, tz_back, bx, by_north, bz_back),
                     dMom, smem);
    }
    if (ty == BLOCK_THREAD_Y - 1 && tz == BLOCK_THREAD_Z - 1 && tx > 0 && tx < BLOCK_THREAD_X - 1)
    {
        // NORTH-FRONT
        load_moments(sx, sy + HALO, sz + HALO,
                     IDX_BLOCK(tx, ty_north, tz_front, bx, by_north, bz_front),
                     dMom, smem);
    }
    __syncthreads();
    // ========================================= CORNERS ==============================================
    // Bottom-back (z - HALO)
    if (tx == 0 && ty == 0 && tz == 0)
    {
        // WEST-SOUTH-BACK
        load_moments(sx - HALO, sy - HALO, sz - HALO,
                     IDX_BLOCK(tx_west, ty_south, tz_back, bx_west, by_south, bz_back),
                     dMom, smem);
    }
    if (tx == 0 && ty == 0 && tz == BLOCK_THREAD_Z - 1)
    {
        // WEST-SOUTH-FRONT
        load_moments(sx - HALO, sy - HALO, sz + HALO,
                     IDX_BLOCK(tx_west, ty_south, tz_front, bx_west, by_south, bz_front),
                     dMom, smem);
    }
    if (tx == 0 && ty == BLOCK_THREAD_Y - 1 && tz == 0)
    {
        // WEST-NORTH-BACK
        load_moments(sx - HALO, sy + HALO, sz - HALO,
                     IDX_BLOCK(tx_west, ty_north, tz_back, bx_west, by_north, bz_back),
                     dMom, smem);
    }
    if (tx == 0 && ty == BLOCK_THREAD_Y - 1 && tz == BLOCK_THREAD_Z - 1)
    {
        // WEST-NORTH-FRONT
        load_moments(sx - HALO, sy + HALO, sz + HALO,
                     IDX_BLOCK(tx_west, ty_north, tz_front, bx_west, by_north, bz_front),
                     dMom, smem);
    }

    // EAST SIDE
    if (tx == BLOCK_THREAD_X - 1 && ty == 0 && tz == 0)
    {
        // EAST-SOUTH-BACK
        load_moments(sx + HALO, sy - HALO, sz - HALO,
                     IDX_BLOCK(tx_east, ty_south, tz_back, bx_east, by_south, bz_back),
                     dMom, smem);
    }
    if (tx == BLOCK_THREAD_X - 1 && ty == 0 && tz == BLOCK_THREAD_Z - 1)
    {
        // EAST-SOUTH-FRONT
        load_moments(sx + HALO, sy - HALO, sz + HALO,
                     IDX_BLOCK(tx_east, ty_south, tz_front, bx_east, by_south, bz_front),
                     dMom, smem);
    }
    if (tx == BLOCK_THREAD_X - 1 && ty == BLOCK_THREAD_Y - 1 && tz == 0)
    {
        // EAST-NORTH-BACK
        load_moments(sx + HALO, sy + HALO, sz - HALO,
                     IDX_BLOCK(tx_east, ty_north, tz_back, bx_east, by_north, bz_back),
                     dMom, smem);
    }
    if (tx == BLOCK_THREAD_X - 1 && ty == BLOCK_THREAD_Y - 1 && tz == BLOCK_THREAD_Z - 1)
    {
        // EAST-NORTH-FRONT
        load_moments(sx + HALO, sy + HALO, sz + HALO,
                     IDX_BLOCK(tx_east, ty_north, tz_front, bx_east, by_north, bz_front),
                     dMom, smem);
    }
    __syncthreads();
}
#endif // MLBM_H