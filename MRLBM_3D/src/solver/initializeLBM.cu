#include <iostream>
#include <fstream>

#include "initializeLBM.cuh"

__constant__ real d_w[Q];
__constant__ int d_cx[Q];
__constant__ int d_cy[Q];
__constant__ int d_cz[Q];

__constant__ real d_Hxx[Q];
__constant__ real d_Hyy[Q];
__constant__ real d_Hzz[Q];
__constant__ real d_Hxy[Q];
__constant__ real d_Hxz[Q];
__constant__ real d_Hyz[Q];

__device__ real d_sumUx;
__device__ real d_UCONV;

// A simple hash-based random number generator for CUDA
__device__ real get_noise(size_t idx, real scale)
{
    // Basic LCG / Hash to get a pseudo-random float between -1 and 1
    unsigned int seed = (unsigned int)idx;
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);

    // Normalize to [-1.0, 1.0]
    float r = (float)(seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    return scale * (toReal(2.0) * (real)r - toReal(1.0));
}

__global__ void gpu_initialize_Moments_GhostInterface(nodeVar dMom, haloData gHalo)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

    // bounds check
    if (x >= NX || y >= NY || z >= NZ)
        return;

    const size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

    // real rho = RHO_0;
    // real ux = toReal(0.0);
    // real uy = toReal(0.0);
    // real uz = toReal(0.0);

    const real noise_scale = toReal(1e-4) * U_MAX;
    real rho = RHO_0;
    real ux = get_noise(idx, noise_scale);
    real uy = get_noise(idx + 12345, noise_scale);
    real uz = get_noise(idx + 67890, noise_scale);

    real mxx, myy, mzz, mxy, mxz, myz;
    real pop[Q];

    //========================== Initialize nodeTypes and Moments=============================================
    const real umag = ux * ux + uy * uy + uz * uz;
    for (int q = 0; q < Q; q++)
    {
        const real cx = toReal(d_cx[q]);
        const real cy = toReal(d_cy[q]);
        const real cz = toReal(d_cz[q]);

        const real udotc = ux * cx + uy * cy + uz * cz;

        // Equlibrium populations
        pop[q] = d_w[q] * rho * (toReal(1.0) + as2 * udotc + toReal(0.5) * as2 * as2 * udotc * udotc - toReal(0.5) * as2 * umag);
    }
    const real inv_rho = toReal(1.0) / rho;

    mxx = 0.0, myy = 0.0, mzz = 0.0;
    mxy = 0.0, mxz = 0.0, myz = 0.0;
    for (int q = 0; q < Q; q++)
    {
        mxx += pop[q] * d_Hxx[q];
        myy += pop[q] * d_Hyy[q];
        mzz += pop[q] * d_Hzz[q];
        mxy += pop[q] * d_Hxy[q];
        mxz += pop[q] * d_Hxz[q];
        myz += pop[q] * d_Hyz[q];
    }
    mxx *= inv_rho;
    myy *= inv_rho;
    mzz *= inv_rho;
    mxy *= inv_rho;
    mxz *= inv_rho;
    myz *= inv_rho;

    //=================== Writing moments to global memory======================================

    dMom.nodeType[idx] = encodeNode(NODE_BULK, 0);
    dMom.rho[idx] = rho - RHO_0;
    dMom.ux[idx] = ux;
    dMom.uy[idx] = uy;
    dMom.uz[idx] = uz;
    dMom.mxx[idx] = mxx;
    dMom.myy[idx] = myy;
    dMom.mzz[idx] = mzz;
    dMom.mxy[idx] = mxy;
    dMom.mxz[idx] = mxz;
    dMom.myz[idx] = myz;

    //========================== Halo Interface =============================================
    pop_reconstruction(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);

    const unsigned int tx = threadIdx.x; // local thread x id
    const unsigned int ty = threadIdx.y; // local thread y id
    const unsigned int tz = threadIdx.z; // local thread z id
    const unsigned int bx = blockIdx.x;  // local block x id
    const unsigned int by = blockIdx.y;  // local block y id
    const unsigned int bz = blockIdx.z;  // local block z id

    pop_save_to_halo(gHalo, tx, ty, tz, bx, by, bz, pop);
}
