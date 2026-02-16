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

__global__ void streaming_and_evaluate_Mom(const cylinderVar cylinder, nodeVar fMom,
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

    __shared__ real s_pop[THREADS_PER_BLOCK * (Q - 1)]; // allocate populations except stationay population in a block

    // Loading moments from the global memory
    const size_t idx = IDX_BLOCK(tx, ty, tz, bx, by, bz);

    nodeType_t nodeType = fMom.nodeType[idx];
    real rho = RHO_0 + fMom.rho[idx];
    real ux = fMom.ux[idx];
    real uy = fMom.uy[idx];
    real uz = fMom.uz[idx];
    real mxx = fMom.mxx[idx];
    real myy = fMom.myy[idx];
    real mzz = fMom.mzz[idx];
    real mxy = fMom.mxy[idx];
    real mxz = fMom.mxz[idx];
    real myz = fMom.myz[idx];

    real pop[Q];
    // construct populations from the loaded moments
    pop_reconstruction(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);

    // copy the constructed populations to the shared memory for the streaming
    save_pop(s_pop, pop);

    __syncthreads();

    // STREAMING
    streaming(s_pop, pop);

    // Loading populations from the halo layers to local thread
    pop_load_from_halo(fHalo, tx, ty, tz, bx, by, bz, pop);

    //========================== Moments evaluation ========================================
    if (nodeType == BULK)
    {
        //    printf("oKKK::%d\n", toInt(nodeType));
        evaluate_moments(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);
    }
    else
    {
        // if (nodeType >= INNER_NODE && nodeType < (INNER_NODE + d_NB))
        // {
        //     const nodeType_t nodeTag = nodeType - INNER_NODE;
        //     if (rotated_coordinates)
        //     {
        //         evaluate_incoming_moments_rotated(x, y, nodeTag, cylinder, pop, rho, mxx, myy, mxy);
        //     }
        //     else
        //     {
        //         evaluate_incoming_moments(nodeTag, cylinder, pop, rho, mxx, myy, mxy);
        //     }
        // }
        // else if (triangular && nodeType >= (BCFLUID_NODE + 0) && nodeType < (BCFLUID_NODE + 4))
        // {
        //     const nodeType_t nodeTag = nodeType - BCFLUID_NODE;
        //     fluid_boundary_condition(nodeTag, pop, rho, ux, uy, mxx, myy, mxy);
        // }
        // else
        // {
        //     boundary_condition(nodeType, fMom, pop, rho, ux, uy, mxx, myy, mxy);
        // }

        boundary_condition(nodeType, fMom, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }

    __syncthreads();

    fMom.rho[idx] = rho - RHO_0; // Incoming density rhoI only for cylinder boundary nodes
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;
    fMom.uz[idx] = uz;
    fMom.mxx[idx] = mxx; // Incoming Moment mxxI only for cylinder boundary nodes
    fMom.myy[idx] = myy; // Incoming Moment myyI only for cylinder boundary nodes
    fMom.mzz[idx] = mzz; // Incoming Moment mzzI only for cylinder boundary nodes
    fMom.mxy[idx] = mxy; // Incoming Moment mxyI only for cylinder boundary nodes
    fMom.mxz[idx] = mxz; // Incoming Moment mxzI only for cylinder boundary nodes
    fMom.myz[idx] = myz; // Incoming Moment myzI only for cylinder boundary nodes
}

__global__ void collision_halo_update(const cylinderVar cylinder, nodeVar fMom, haloData fHalo, haloData gHalo, const int iter)
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
    // const nodeType_t nodeType = fMom.nodeType[idx];
    real rho = RHO_0 + fMom.rho[idx];
    real ux = fMom.ux[idx];
    real uy = fMom.uy[idx];
    real uz = fMom.uz[idx];
    real mxx = fMom.mxx[idx];
    real myy = fMom.myy[idx];
    real mzz = fMom.mzz[idx];
    real mxy = fMom.mxy[idx];
    real mxz = fMom.mxz[idx];
    real myz = fMom.myz[idx];

    real pop[Q];

    // Collision on moment space
    moment_collision(ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);

    // Regularized populations using post-collisional moments
    pop_reconstruction(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);

    // updating halo interface with this regularized populations
    pop_save_to_halo(gHalo, tx, ty, tz, bx, by, bz, pop);

    // writing Post-collisional moments into global memory
    fMom.rho[idx] = rho - RHO_0;
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;
    fMom.uz[idx] = uz;
    fMom.mxx[idx] = mxx;
    fMom.myy[idx] = myy;
    fMom.mzz[idx] = mzz;
    fMom.mxy[idx] = mxy;
    fMom.mxz[idx] = mxz;
    fMom.myz[idx] = myz;
}
