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
__constant__ int d_NB_FLUID;
__constant__ int d_NB_SOLID;

__device__ real d_sumUx;
__device__ real d_UCONV;

__device__ real d_TotalFx;
__device__ real d_TotalFy;
__device__ real d_TotalFz;
__device__ real d_Totalm;

__constant__ uint32_t d_incomingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_incomingMask_bcsolid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcsolid[MAX_NODE_TAG];

#endif

void initialize_domain(nodeVar &dMom, nodeVar &hMom, haloData &gHalo, cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    gpu_initialize_Moments_nodeType_GhostInterface<<<grid, block>>>(dMom, gHalo);
    checkKernelExecution();

    initialize_nodeType(hMom);

#ifdef CYLINDER
    triangular ? initialize_cylinder_nodeType_triangular(hMom) : initialize_cylinder_nodeType_staircase(hMom);
    write_geometry_files(hMom);

    allocateCylinderMemory(h_cylinder, d_cylinder);
    buildBoundaryList_updateBoundaryNodeType(hMom, h_cylinder);

    find_incomings_outgoings(hMom, h_cylinder.boundaryList, h_cylinder.incomingMask, h_cylinder.outgoingMask, NB);
    setup_bcfluid_masks(hMom, h_cylinder);
    setup_bcsolid_masks(hMom, h_cylinder);

    copyHostToDevice(d_cylinder, h_cylinder);
#endif
    initialize_host_device_constants();
}

__global__ void gpu_initialize_Moments_nodeType_GhostInterface(nodeVar dMom, haloData gHalo)
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
    const size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, threadIdx.z,
                                 blockIdx.x, blockIdx.y, blockIdx.z);
    dMom.nodeType[idx] = BULK;
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
