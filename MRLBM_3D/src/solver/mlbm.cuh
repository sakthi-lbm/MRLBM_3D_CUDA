#ifndef MLBM_H
#define MLBM_H

#include "solver/initializeLBM.cuh"
#include HALO_INTERFACE
#include STREAMING
#include EVAL_MOMENTS

__global__ void streaming_and_evaluate_Mom(const cylinderVar cylinder, nodeVar fMom,
                                           haloData fHalo, haloData gHalo, const int iter);
__global__ void collision_halo_update(const cylinderVar cylinder, nodeVar fMom,
                                      haloData fHalo, haloData gHalo, const int iter);

__device__ __forceinline__ void load_moments(const unsigned int sx,
                                             const unsigned int sy,
                                             const unsigned int sz,
                                             const size_t gidx,
                                             const nodeVar &dMom, moments &smem)
{
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
    const unsigned int x = threadIdx.x + blockIdx.x * BLOCK_THREAD_X;
    const unsigned int y = threadIdx.y + blockIdx.y * BLOCK_THREAD_Y;
    const unsigned int z = threadIdx.z + blockIdx.z * BLOCK_THREAD_Z;

    const unsigned int tx = threadIdx.x + HALO;
    const unsigned int ty = threadIdx.y + HALO;
    const unsigned int tz = threadIdx.z + HALO;

    size_t gidx;
    // ---------------- interior ----------------
    gidx = IDX(x, y, z);
    load_moments(tx, ty, tz, gidx, dMom, smem);

    int x_west = x - HALO;
    if (x_west < 0)
        x_west += NX;
    int x_east = x + BLOCK_THREAD_X;
    if (x_east >= NX)
        x_east -= NX;

    int y_south = y - HALO;
    if (y_south < 0)
        y_south += NY;
    int y_north = y + BLOCK_THREAD_Y;
    if (y_north >= NY)
        y_north -= NY;

    int z_back = z - HALO;
    if (z_back < 0)
        z_back += NZ;
    int z_front = z + BLOCK_THREAD_Z;
    if (z_front >= NZ)
        z_front -= NZ;
    // ========================================= FACE HALOS ==============================================
    if (threadIdx.x < HALO)
    {
        gidx = IDX(x_west, y, z);
        load_moments(tx - HALO, ty, tz, gidx, dMom, smem);

        gidx = IDX(x_east, y, z);
        load_moments(tx + BLOCK_THREAD_X, ty, tz, gidx, dMom, smem);
    }

    if (threadIdx.y < HALO)
    {
        gidx = IDX(x, y_south, z);
        load_moments(tx, ty - HALO, tz, gidx, dMom, smem);

        gidx = IDX(x, y_north, z);
        load_moments(tx, ty + BLOCK_THREAD_Y, tz, gidx, dMom, smem);
    }

    if (threadIdx.z < HALO)
    {
        gidx = IDX(x, y, z_back);
        load_moments(tx, ty, tz - HALO, gidx, dMom, smem);

        gidx = IDX(x, y, z_front);
        load_moments(tx, ty, tz + BLOCK_THREAD_Z, gidx, dMom, smem);
    }

    // ========================================= EDGE HALOS ==============================================
    if (threadIdx.x < HALO && threadIdx.y < HALO)
    {
        // 4 edges in XY plane
        load_moments(tx - HALO, ty - HALO, tz, IDX(x_west, y_south, z), dMom, smem);
        load_moments(tx + BLOCK_THREAD_X, ty - HALO, tz, IDX(x_east, y_south, z), dMom, smem);
        load_moments(tx - HALO, ty + BLOCK_THREAD_Y, tz, IDX(x_west, y_north, z), dMom, smem);
        load_moments(tx + BLOCK_THREAD_X, ty + BLOCK_THREAD_Y, tz, IDX(x_east, y_north, z), dMom, smem);
    }

    if (threadIdx.x < HALO && threadIdx.z < HALO)
    {
        // 4 edges in XZ plane
        load_moments(tx - HALO, ty, tz - HALO, IDX(x_west, y, z_back), dMom, smem);
        load_moments(tx + BLOCK_THREAD_X, ty, tz - HALO, IDX(x_east, y, z_back), dMom, smem);
        load_moments(tx - HALO, ty, tz + BLOCK_THREAD_Z, IDX(x_west, y, z_front), dMom, smem);
        load_moments(tx + BLOCK_THREAD_X, ty, tz + BLOCK_THREAD_Z, IDX(x_east, y, z_front), dMom, smem);
    }

    if (threadIdx.y < HALO && threadIdx.z < HALO)
    {

        // 4 edges in YZ plane
        load_moments(tx, ty - HALO, tz - HALO, IDX(x, y_south, z_back), dMom, smem);
        load_moments(tx, ty + BLOCK_THREAD_Y, tz - HALO, IDX(x, y_north, z_back), dMom, smem);
        load_moments(tx, ty - HALO, tz + BLOCK_THREAD_Z, IDX(x, y_south, z_front), dMom, smem);
        load_moments(tx, ty + BLOCK_THREAD_Y, tz + BLOCK_THREAD_Z, IDX(x, y_north, z_front), dMom, smem);
    }

    // ========================================= CORNERS ==============================================
    if (threadIdx.x < HALO && threadIdx.y < HALO && threadIdx.z < HALO)
    {
        // Bottom-back
        load_moments(tx - HALO, ty - HALO, tz - HALO, IDX(x_west, y_south, z_back), dMom, smem);
        load_moments(tx + BLOCK_THREAD_X, ty - HALO, tz - HALO, IDX(x_east, y_south, z_back), dMom, smem);
        load_moments(tx - HALO, ty + BLOCK_THREAD_Y, tz - HALO, IDX(x_west, y_north, z_back), dMom, smem);
        load_moments(tx + BLOCK_THREAD_X, ty + BLOCK_THREAD_Y, tz - HALO, IDX(x_east, y_north, z_back), dMom, smem);

        // Top-front
        load_moments(tx - HALO, ty - HALO, tz + BLOCK_THREAD_Z, IDX(x_west, y_south, z_front), dMom, smem);
        load_moments(tx + BLOCK_THREAD_X, ty - HALO, tz + BLOCK_THREAD_Z, IDX(x_east, y_south, z_front), dMom, smem);
        load_moments(tx - HALO, ty + BLOCK_THREAD_Y, tz + BLOCK_THREAD_Z, IDX(x_west, y_north, z_front), dMom, smem);
        load_moments(tx + BLOCK_THREAD_X, ty + BLOCK_THREAD_Y, tz + BLOCK_THREAD_Z, IDX(x_east, y_north, z_front), dMom, smem);
    }
}
#endif // MLBM_H