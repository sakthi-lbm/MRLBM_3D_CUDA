#include <iostream>
#include <fstream>

#include "initializeLBM.cuh"

#ifdef CYLINDER
__device__ __constant__ int d_NB;
__device__ __constant__ int d_NBCF;

__device__ __constant__ binary_t d_incomings_bcfluid[4][Q];
__device__ __constant__ binary_t d_outgoings_bcfluid[4][Q];

__device__ __constant__ real d_Hxx[Q];
__device__ __constant__ real d_Hyy[Q];
__device__ __constant__ real d_Hxy[Q];

__device__ real d_TotalFx;
__device__ real d_TotalFy;
__device__ real d_Totalm;

#endif

void initialize_domain(nodeVar &dMom, nodeVar &hMom, haloData &gHalo, cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    gpu_initialize_Moments_nodeType_GhostInterface<<<grid, block>>>(dMom, gHalo);
    checkKernelExecution();

    initialize_nodeType(hMom);
    triangular ? initialize_cylinder_nodeType_triangular(hMom) : initialize_cylinder_nodeType(hMom);

    initialize_host_device_constants();
    write_geometry_files(hMom);

    allocateCylinderMemory(h_cylinder, d_cylinder, NB);
    buildBoundaryList_updateBoundaryNodeType(hMom, h_cylinder);

    find_incomings_outgoings_cylinder(hMom, h_cylinder, NB);
    find_incoming_outgoings_bcfluid();

    copyHostToDevice(d_cylinder, h_cylinder, NB);
}

__global__ void gpu_initialize_Moments_nodeType_GhostInterface(nodeVar fMom, haloData gHalo)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    // bounds check
    if (x >= NX || y >= NY)
        return;

    real rho = RHO_0;
    real ux = toReal(0.0);
    real uy = toReal(0.0);

    real mxx, myy, mxy;

    const size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    //========================== Initialize nodeTypes and Moments=============================================

    fMom.nodeType[idx] = BULK;
    fMom.rho[idx] = rho - RHO_0;
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;

    real pop[Q];
    for (int i = 0; i < Q; i++)
    {
        real umag = ux * ux + uy * uy;
        real udotc = ux * d_cx[i] + uy * d_cy[i];

        // Equlibrium populations
        pop[i] = w[i] * rho * (toReal(1.0) + as2 * udotc + toReal(0.5) * as2 * as2 * udotc * udotc - toReal(0.5) * as2 * umag);
    }
    const real inv_rho = toReal(1.0) / rho;
    fMom.mxx[idx] = (pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[8]) * inv_rho - cs2;
    fMom.myy[idx] = (pop[2] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]) * inv_rho - cs2;
    fMom.mxy[idx] = (pop[5] - pop[6] + pop[7] - pop[8]) * inv_rho;

    //========================== Halo Interface =============================================
    rho = RHO_0 + fMom.rho[idx];
    ux = fMom.ux[idx];
    uy = fMom.uy[idx];
    mxx = fMom.mxx[idx];
    myy = fMom.myy[idx];
    mxy = fMom.mxy[idx];

    pop_reconstruction(rho, ux, uy, mxx, myy, mxy, pop);

    const unsigned int tx = threadIdx.x; // local thread x id
    const unsigned int ty = threadIdx.y; // local thread y id
    const unsigned int bx = blockIdx.x;  // local block x id
    const unsigned int by = blockIdx.y;  // local block y id

    pop_save_to_halo(gHalo, tx, ty, bx, by, pop);
}
