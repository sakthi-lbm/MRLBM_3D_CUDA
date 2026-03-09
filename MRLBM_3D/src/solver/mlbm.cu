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

__global__ void streaming_and_evaluate_Mom(cylinderVar cylinder, nodeVar dMom,
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

    __shared__ real s_pop[THREADS_PER_BLOCK * Q]; // allocate populations except stationay population in a block

    // Loading moments from the global memory
    const size_t idx = IDX_BLOCK(tx, ty, tz, bx, by, bz);

    nodeType_t nodeType = dMom.nodeType[idx];
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

    // updating shared memory pop with streamed populations for neumann condition
    // if constexpr (NEUMANN_CURRENT_UPDATE)
    // {
    //     save_pop(s_pop, pop);
    //     __syncthreads();
    // }

    //========================== Moments evaluation ========================================
    if (nodeType == BULK)
    {
        evaluate_moments(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);
    }
    else
    {
        if (nodeType >= INNER_NODE && nodeType < (INNER_NODE + d_NB))
        {
            const nodeType_t nodeTag = nodeType - INNER_NODE;
            evaluate_incoming_moments_rotated(x, y, z, nodeTag, cylinder, pop, rho, mxx, myy, mzz, mxy, mxz, myz);
        }
        else
        {
            boundary_condition(nodeType, dMom, pop, s_pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        }
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

__global__ void apply_bc_cylinder(const int NB, const nodeType_t NODE_TYPE, const cylinderVar &cylinder,
                                  nodeVar dMom, const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                  const real D_WALL, const int iter)
{

    // boundary index
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= NB)
        return;

    // global index loaded from the boundary list
    const size_t idx = cylinder.boundaryList[i];

    // converting global index to the gloabl coordinates
    unsigned int x, y, z;
    GlobalIndexToXYZ(idx, x, y, z);

    const nodeType_t nodeType = dMom.nodeType[idx];
    real rho = RHO_0 + dMom.rho[idx]; // Incoming density rhoI
    real ux = dMom.ux[idx];
    real uy = dMom.uy[idx];
    real uz = dMom.uz[idx];
    real mxx = dMom.mxx[idx]; // Incoming Moment mxxI
    real myy = dMom.myy[idx]; // Incoming Moment myyI
    real mzz = dMom.mzz[idx]; // Incoming Moment mzzI
    real mxy = dMom.mxy[idx]; // Incoming Moment mxyI
    real mxz = dMom.mxz[idx]; // Incoming Moment mxzI
    real myz = dMom.myz[idx]; // Incoming Moment myzI

    // printf("i: %d, x: %d, y: %d, nodetype: %d \n", i, x, y, nodeType);

    if (nodeType >= NODE_TYPE && nodeType < (NODE_TYPE + NB))
    {
        cylinder_boundary_condition_rotated(x, y, z, cylinder, nodeType, dMom, rho, ux, uy, uz,
                                            mxx, myy, mzz, mxy, mxz, myz, UX_PRIME, UY_PRIME, UZ_PRIME,
                                            NODE_TYPE, D_WALL, iter);
    }

    // writing  moments into global memory (being done only for cylinder block)
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

    // Regularized populations using post-collisional moments
    real pop[Q];
    pop_reconstruction(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);

    // updating halo interface with this regularized populations
    pop_save_to_halo(gHalo, tx, ty, tz, bx, by, bz, pop);

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
