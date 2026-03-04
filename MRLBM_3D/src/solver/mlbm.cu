#include "mlbm.cuh"

__device__ void inline moment_collision(const real ux, const real uy, const real uz,
                                        real &mxx, real &myy, real &mzz,
                                        real &mxy, real &mxz, real &myz)
{
    const real omegaVar = OMEGA;
    const real omega_m1 = toReal(1.0) - omegaVar;

    mxx = omega_m1 * mxx + omegaVar * ux * ux;
    myy = omega_m1 * myy + omegaVar * uy * uy;
    mzz = omega_m1 * mzz + omegaVar * uz * uz;
    mxy = omega_m1 * mxy + omegaVar * ux * uy;
    mxz = omega_m1 * mxz + omegaVar * ux * uz;
    myz = omega_m1 * myz + omegaVar * uy * uz;
}

__global__ void streaming_and_evaluate_Mom(const cylinderVar cylinder, nodeVar dMom,
                                           haloData fHalo, haloData gHalo, const int iter)
{

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int bz = blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    real rho;
    real ux, uy, uz;
    real mxx, myy, mzz, mxy, mxz, myz;

    const size_t idx = IDX_BLOCK(tx, ty, tz, bx, by, bz);
    const nodeType_t nodetype = dMom.nodeType[idx];

    __shared__ moments s_mom;

    load_shared_moments(dMom, s_mom);
    __syncthreads();

    // STREAMING
    real pop[Q];
    streaming(s_mom, pop);

    //========================== Moments evaluation ========================================
    if (nodetype == BULK)
    {
        evaluate_moments(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);
    }
    else
    {
        boundary_condition(nodetype, dMom, pop, s_mom, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }

    dMom.rho[idx] = rho - RHO_0; // Incoming density rhoI only for cylinder boundary nodes
    dMom.ux[idx] = ux;
    dMom.uy[idx] = uy;
    dMom.uz[idx] = uz;
    dMom.mxx[idx] = mxx; // Incoming Moment mxxI only for cylinder boundary nodes
    dMom.myy[idx] = myy; // Incoming Moment myyI only for cylinder boundary nodes
    dMom.mzz[idx] = mzz; // Incoming Moment mzzI only for cylinder boundary nodes
    dMom.mxy[idx] = mxy; // Incoming Moment mxyI only for cylinder boundary nodes
    dMom.mxz[idx] = mxz; // Incoming Moment mxzI only for cylinder boundary nodes
    dMom.myz[idx] = myz; // Incoming Moment myzI only for cylinder boundary nodes
}

__global__ void collision_halo_update(const cylinderVar cylinder, nodeVar dMom, haloData fHalo, haloData gHalo, const int iter)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int bz = blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    // Loading moments from the global memory
    const size_t idx = IDX_BLOCK(tx, ty, tz, bx, by, bz);
    // const nodeType_t nodeType = dMom.nodeType[idx];
    real rho = RHO_0 + dMom.rho[idx];
    real ux = dMom.ux[idx];
    real uy = dMom.uy[idx];
    real uz = dMom.uz[idx];
    real mxx = dMom.mxx[idx];
    real myy = dMom.myy[idx];
    real mzz = dMom.mzz[idx];
    real mxy = dMom.mxy[idx];
    real mxz = dMom.mxz[idx];
    real myz = dMom.myz[idx];

    // Collision on moment space
    moment_collision(ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);

    // writing Post-collisional moments into global memory
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
