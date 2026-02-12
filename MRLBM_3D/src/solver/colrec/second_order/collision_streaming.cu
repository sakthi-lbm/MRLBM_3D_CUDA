#include <iostream>
#include "collision_streaming.cuh"

__device__ void mom_collision(real ux, real uy, real &mxx, real &myy, real &mxy)
{
    const real omegaVar = OMEGA;
    const real omega_m1 = 1.0 - omegaVar;

    mxx = omega_m1 * mxx + omegaVar * ux * ux;
    myy = omega_m1 * myy + omegaVar * uy * uy;
    mxy = omega_m1 * mxy + omegaVar * ux * uy;
}

__global__ void streaming_and_evaluate_Mom(const cylinderVar cylinder, nodeVar fMom,
                                           haloData fHalo, haloData gHalo, const int iter)
{

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;

    if (x >= NX || y >= NY)
        return;

    __shared__ real s_pop[THREADS_PER_BLOCK * (Q - 1)]; // allocate populations except stationay population in a block

    // Loading moments from the global memory
    const size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    nodeType_t nodeType = fMom.nodeType[idx];
    real rho = RHO_0 + fMom.rho[idx];
    real ux = fMom.ux[idx];
    real uy = fMom.uy[idx];
    real mxx = fMom.mxx[idx];
    real myy = fMom.myy[idx];
    real mxy = fMom.mxy[idx];

    // printf("node: %d\n", fMom.nodeType[IDX(0, NY - 1)]);

    real pop[Q];
    // construct populations from the loaded moments
    pop_reconstruction(rho, ux, uy, mxx, myy, mxy, pop);

    // copy the constructed populations to the shared memory for the streaming
    s_pop[idxPopBlock(tx, ty, 0)] = pop[1];
    s_pop[idxPopBlock(tx, ty, 1)] = pop[2];
    s_pop[idxPopBlock(tx, ty, 2)] = pop[3];
    s_pop[idxPopBlock(tx, ty, 3)] = pop[4];
    s_pop[idxPopBlock(tx, ty, 4)] = pop[5];
    s_pop[idxPopBlock(tx, ty, 5)] = pop[6];
    s_pop[idxPopBlock(tx, ty, 6)] = pop[7];
    s_pop[idxPopBlock(tx, ty, 7)] = pop[8];

    __syncthreads();

    // STREAMING
    const unsigned int xm1 = (threadIdx.x - 1 + blockDim.x) % blockDim.x;
    const unsigned int xp1 = (threadIdx.x + 1 + blockDim.x) % blockDim.x;
    const unsigned int ym1 = (threadIdx.y - 1 + blockDim.y) % blockDim.y;
    const unsigned int yp1 = (threadIdx.y + 1 + blockDim.y) % blockDim.y;

    pop[1] = s_pop[idxPopBlock(xm1, ty, 0)];
    pop[2] = s_pop[idxPopBlock(tx, ym1, 1)];
    pop[3] = s_pop[idxPopBlock(xp1, ty, 2)];
    pop[4] = s_pop[idxPopBlock(tx, yp1, 3)];
    pop[5] = s_pop[idxPopBlock(xm1, ym1, 4)];
    pop[6] = s_pop[idxPopBlock(xp1, ym1, 5)];
    pop[7] = s_pop[idxPopBlock(xp1, yp1, 6)];
    pop[8] = s_pop[idxPopBlock(xm1, yp1, 7)];

    // Loading populations from the halo layers to local thread
    pop_load_from_halo(fHalo, tx, ty, bx, by, pop);

    //========================== Moments evaluation ========================================
    if (nodeType == BULK)
    {
        rho = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8];
        const real invRho = 1.0 / rho;

        ux = (pop[1] - pop[3] + pop[5] - pop[6] - pop[7] + pop[8]) * invRho;
        uy = (pop[2] - pop[4] + pop[5] + pop[6] - pop[7] - pop[8]) * invRho;

        mxx = (pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[8]) * invRho - cs2;
        myy = (pop[2] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]) * invRho - cs2;
        mxy = (pop[5] - pop[6] + pop[7] - pop[8]) * invRho;
    }
    else
    {
        if (nodeType >= INNER_NODE && nodeType < (INNER_NODE + d_NB))
        {
            const nodeType_t nodeTag = nodeType - INNER_NODE;
            if (rotated_coordinates)
            {
                evaluate_incoming_moments_rotated(x, y, nodeTag, cylinder, pop, rho, mxx, myy, mxy);
            }
            else
            {
                evaluate_incoming_moments(nodeTag, cylinder, pop, rho, mxx, myy, mxy);
            }
        }
        else if (triangular && nodeType >= (BCFLUID_NODE + 0) && nodeType < (BCFLUID_NODE + 4))
        {
            const nodeType_t nodeTag = nodeType - BCFLUID_NODE;
            fluid_boundary_condition(nodeTag, pop, rho, ux, uy, mxx, myy, mxy);
        }
        else
        {
            boundary_condition(nodeType, fMom, pop, rho, ux, uy, mxx, myy, mxy);
        }
    }

    fMom.rho[idx] = rho - RHO_0; // Incoming density rhoI only for cylinder boundary nodes
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;
    fMom.mxx[idx] = mxx; // Incoming Moment mxxI only for cylinder boundary nodes
    fMom.myy[idx] = myy; // Incoming Moment myyI only for cylinder boundary nodes
    fMom.mxy[idx] = mxy; // Incoming Moment mxyI only for cylinder boundary nodes

    // Incoming force calculation using post-streaming populations
    if (iter >= STAT_START && iter <= STAT_END)
    {
        const real rx = toReal(x) - XC;
        const real ry = toReal(y) - YC;

        real Fx_local = 0.0;
        real Fy_local = 0.0;
        real m_local = 0.0;
        if (nodeType >= INNER_NODE && nodeType < INNER_NODE + d_NB)
        {
            const nodeType_t nodeTag = nodeType - INNER_NODE;
            compute_incoming_forces_mass(nodeTag, cylinder, pop, Fx_local, Fy_local, m_local);
            atomicAdd(&d_TotalFx, Fx_local);
            atomicAdd(&d_TotalFy, Fy_local);
            atomicAdd(&d_Totalm, m_local);
        }
        else if (nodeType >= BCFLUID_NODE && nodeType < BCFLUID_NODE + 4)
        {
            const nodeType_t nodeTag = nodeType - BCFLUID_NODE;
            compute_incoming_forces_mass_bcfluid(nodeTag, pop, Fx_local, Fy_local, m_local);
            atomicAdd(&d_TotalFx, Fx_local);
            atomicAdd(&d_TotalFy, Fy_local);
            atomicAdd(&d_Totalm, m_local);
        }
    }
}

__global__ void apply_bc_cylinder(const int NB, const int NODE_TYPE, const cylinderVar &cylinder, nodeVar &fMom,
                                  const real UX_PRIME, const real UY_PRIME, const real D_WALL, const int iter)
{

    // boundary index
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= NB)
        return;

    // global index loaded from the boundary list
    const size_t idx = cylinder.boundaryList[i];

    // converting global index to the gloabl coordinates
    unsigned int x, y;
    GlobalIndexToXY(idx, x, y);

    const nodeType_t nodeType = fMom.nodeType[idx];
    real rho = RHO_0 + fMom.rho[idx]; // Incoming density rhoI
    real ux = fMom.ux[idx];
    real uy = fMom.uy[idx];
    real mxx = fMom.mxx[idx]; // Incoming Moment mxxI
    real myy = fMom.myy[idx]; // Incoming Moment myyI
    real mxy = fMom.mxy[idx]; // Incoming Moment mxyI

    // printf("i: %d, x: %d, y: %d, nodetype: %d \n", i, x, y, nodeType);

    if (nodeType >= NODE_TYPE && nodeType < (NODE_TYPE + NB))
    {
        if (rotated_coordinates)
        {
            cylinder_boundary_condition_rotated(x, y, cylinder, nodeType, fMom, rho, ux, uy, mxx, myy, mxy,
                                                UX_PRIME, UY_PRIME, NODE_TYPE, D_WALL, iter);
        }
        else
        {
            cylinder_boundary_condition(x, y, cylinder, nodeType, fMom, rho, ux, uy, mxx, myy, mxy,
                                        UX_PRIME, UY_PRIME, NODE_TYPE, D_WALL);
        }
    }

    // writing  moments into global memory (being done only for cylinder block)
    fMom.rho[idx] = rho - RHO_0;
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;
    fMom.mxx[idx] = mxx;
    fMom.myy[idx] = myy;
    fMom.mxy[idx] = mxy;
}

__global__ void collision_halo_update(const cylinderVar cylinder, nodeVar fMom, haloData fHalo, haloData gHalo, const int iter)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;

    if (x >= NX || y >= NY)
        return;

    // Loading moments from the global memory
    const size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    const nodeType_t nodeType = fMom.nodeType[idx];
    real rho = RHO_0 + fMom.rho[idx];
    real ux = fMom.ux[idx];
    real uy = fMom.uy[idx];
    real mxx = fMom.mxx[idx];
    real myy = fMom.myy[idx];
    real mxy = fMom.mxy[idx];

    real pop[Q];

    // Collision on moment space
    mom_collision(ux, uy, mxx, myy, mxy);

    // Regularized populations using post-collisional moments
    pop_reconstruction(rho, ux, uy, mxx, myy, mxy, pop);

    // updating halo interface with this regularized populations
    pop_save_to_halo(gHalo, tx, ty, bx, by, pop);

    // writing Post-collisional moments into global memory
    fMom.rho[idx] = rho - RHO_0;
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;
    fMom.mxx[idx] = mxx;
    fMom.myy[idx] = myy;
    fMom.mxy[idx] = mxy;

    // Outgoing force calculation using post-streaming populations
    if (iter >= STAT_START && iter <= STAT_END)
    {
        const real rx = toReal(x) - XC;
        const real ry = toReal(y) - YC;

        real Fx_local = 0.0;
        real Fy_local = 0.0;
        real m_local = 0.0;
        if (nodeType >= INNER_NODE && nodeType < INNER_NODE + d_NB)
        {
            const nodeType_t nodeTag = nodeType - INNER_NODE;
            compute_outgoing_forces_mass(nodeTag, cylinder, pop, Fx_local, Fy_local, m_local);
            atomicAdd(&d_TotalFx, -Fx_local);
            atomicAdd(&d_TotalFy, -Fy_local);
            atomicAdd(&d_Totalm, -m_local);
        }
        else if (nodeType >= BCFLUID_NODE && nodeType < (BCFLUID_NODE + 4))
        {
            const nodeType_t nodeTag = nodeType - BCFLUID_NODE;
            compute_outgoing_forces_mass_bcfluid(nodeTag, pop, Fx_local, Fy_local, m_local);
            atomicAdd(&d_TotalFx, -Fx_local);
            atomicAdd(&d_TotalFy, -Fy_local);
            atomicAdd(&d_Totalm, -m_local);
        }
    }
}
