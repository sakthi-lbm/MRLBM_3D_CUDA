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

void initialize_domain(nodeVar &dMom, nodeVar &hMom, cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    gpu_initialize_Moments_nodeType<<<grid, block>>>(dMom);
    checkKernelExecution();

    initialize_nodeType(hMom);
    // initialize_cylinder_nodeType(hMom);
    write_geometry_files(hMom);
}

__global__ void gpu_initialize_Moments_nodeType(nodeVar dMom)
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
        real udotc = ux * d_cx[q] + uy * d_cy[q] + uz * d_cz[q];

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

}
