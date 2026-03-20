#pragma once

#include "../config.h"
#include CASE_KERNALS

__global__ void gpu_initialize_Moments_GhostInterface(nodeVar dMom, haloData gHalo);

inline void check_tau()
{
    if (TAU <= 0.5 || TAU >= 2.5)
    {
        printf("ERROR: Tau out of stable LBM range: tau = %f\n", TAU);
        exit(EXIT_FAILURE);
    }
}

inline void check_mach()
{
    const real Ma = U_MAX / sqrt(cs2); // cs2 = 1/3

    if (Ma > 0.33)
    {
        printf("ERROR: Mach number too high: Ma = %f\n", Ma);
        exit(EXIT_FAILURE);
    }
}

inline void initialize_host_device_constants()
{
    for (int q = 0; q < Q; q++)
    {
        h_Hxx[q] = h_cx[q] * h_cx[q] - cs2;
        h_Hyy[q] = h_cy[q] * h_cy[q] - cs2;
        h_Hzz[q] = h_cz[q] * h_cz[q] - cs2;
        h_Hxy[q] = h_cx[q] * h_cy[q];
        h_Hxz[q] = h_cx[q] * h_cz[q];
        h_Hyz[q] = h_cy[q] * h_cz[q];
    }
    // copy to GPU constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_w, h_w, sizeof(h_w)));
    checkCudaErrors(cudaMemcpyToSymbol(d_cx, h_cx, sizeof(h_cx)));
    checkCudaErrors(cudaMemcpyToSymbol(d_cy, h_cy, sizeof(h_cy)));
    checkCudaErrors(cudaMemcpyToSymbol(d_cz, h_cz, sizeof(h_cz)));

    checkCudaErrors(cudaMemcpyToSymbol(d_Hxx, h_Hxx, sizeof(h_Hxx)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hyy, h_Hyy, sizeof(h_Hyy)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hzz, h_Hzz, sizeof(h_Hzz)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hxy, h_Hxy, sizeof(h_Hxy)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hxz, h_Hxz, sizeof(h_Hxz)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hyz, h_Hyz, sizeof(h_Hyz)));

    check_tau();
    check_mach();
}

inline void initialize_nodeType(nodeVar &hMom)
{
    static_assert(NX % BLOCK_THREAD_X == 0, "NX must tile block size");
    static_assert(NY % BLOCK_THREAD_Y == 0, "NY must tile block size");
    static_assert(NZ % BLOCK_THREAD_Z == 0, "NZ must tile block size");
    for (int z = 0; z < NZ; z++)
    {
        for (int y = 0; y < NY; y++)
        {
            for (int x = 0; x < NX; x++)
            {
                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             z % BLOCK_THREAD_Z,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y,
                                             z / BLOCK_THREAD_Z);
                hMom.nodeType[idx] = boundary_definitions(x, y, z);
            }
        }
    }
}