#include <iostream>
#include <fstream>

#include "initializeLBM.cuh"
#include HALO_INTERFACE

#ifdef CYLINDER
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

__constant__ int d_NB;
__constant__ int d_NBCF;

#endif

void initialize_domain(nodeVar &dMom, nodeVar &hMom, haloData &gHalo, cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    gpu_initialize_Moments_nodeType_GhostInterface<<<grid, block>>>(dMom, gHalo);
    checkKernelExecution();

    initialize_nodeType(hMom);
    write_geometry_files(hMom);
}

__global__ void gpu_initialize_Moments_nodeType_GhostInterface(nodeVar fMom, haloData gHalo)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

    // bounds check
    if (x >= NX || y >= NY || z >= NZ)
        return;

    real rho = RHO_0;
    real ux = toReal(0.0);
    real uy = toReal(0.0);
    real uz = toReal(0.0);

    real mxx, myy, mzz, mxy, mxz, myz;

    const size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, threadIdx.z,
                                 blockIdx.x, blockIdx.y, blockIdx.z);

    //========================== Initialize nodeTypes and Moments=============================================
    fMom.nodeType[idx] = BULK;
    fMom.rho[idx] = rho - RHO_0;
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;
    fMom.uz[idx] = uz;

    real pop[Q];
    for (int i = 0; i < Q; i++)
    {
        real umag = ux * ux + uy * uy + uz * uz;
        real udotc = ux * d_cx[i] + uy * d_cy[i]+ uz * d_cz[i];

        // Equlibrium populations
        pop[i] = d_w[i] * rho * (toReal(1.0) + as2 * udotc + toReal(0.5) * as2 * as2 * udotc * udotc - toReal(0.5) * as2 * umag);
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
    fMom.mxx[idx] = mxx * inv_rho;
    fMom.myy[idx] = myy * inv_rho;
    fMom.mzz[idx] = mzz * inv_rho;
    fMom.mxy[idx] = mxy * inv_rho;
    fMom.mxz[idx] = mxz * inv_rho;
    fMom.myz[idx] = myz * inv_rho;

    //========================== Halo Interface =============================================
    rho = RHO_0 + fMom.rho[idx];
    ux = fMom.ux[idx];
    uy = fMom.uy[idx];
    uz = fMom.uz[idx];
    mxx = fMom.mxx[idx];
    myy = fMom.myy[idx];
    mzz = fMom.mzz[idx];
    mxy = fMom.mxy[idx];
    mxz = fMom.mxz[idx];
    myz = fMom.myz[idx];

    pop_reconstruction(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);

    const unsigned int tx = threadIdx.x; // local thread x id
    const unsigned int ty = threadIdx.y; // local thread y id
    const unsigned int tz = threadIdx.z; // local thread z id
    const unsigned int bx = blockIdx.x;  // local block x id
    const unsigned int by = blockIdx.y;  // local block y id
    const unsigned int bz = blockIdx.z;  // local block z id

    pop_save_to_halo(gHalo, tx, ty, tz, bx, by, bz, pop);
}
