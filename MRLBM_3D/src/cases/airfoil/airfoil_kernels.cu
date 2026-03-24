#include "airfoil_kernels.cuh"

void airfoil_initialize(Simulation &sim)
{
    // allocate struct
    airfoilVar *h_airfoil = new airfoilVar;
    airfoilVar *d_airfoil = new airfoilVar;

    airfoilPostProcess *h_airfoilPost = new airfoilPostProcess;
    airfoilPostProcess *d_airfoilPost = new airfoilPostProcess;

    // compute geometry + counts (NB, etc.)
    triangular ? initialize_airfoil_nodeType_triangular(sim.h_fMom, *h_airfoil)
               : initialize_airfoil_nodeType_staircase(sim.h_fMom, *h_airfoil);

    write_geometry_files(sim.h_fMom);

    // allocate memory using computed sizes
    allocateairfoilMemory(*h_airfoil, *d_airfoil, *h_airfoilPost, *d_airfoilPost);

    buildBoundaryList_updateBoundaryNodeType(sim.h_fMom, *h_airfoil);
    compute_unit_vectors_boundary_nodes(sim.h_fMom, *h_airfoil, D_WALL);
    find_incomings_outgoings(sim.h_fMom, *h_airfoil);
    setup_bcfluid_masks(sim.h_fMom, *h_airfoil);
    setup_bcsolid_masks(sim.h_fMom, *h_airfoil);

    // copy data to device arrays
    copyHostToDevice(*d_airfoil, *h_airfoil);
    airfoil_host_device_constants();

    // store in simulation
    sim.h_caseData = h_airfoil;
    sim.d_caseData = d_airfoil;

    sim.h_casePost = h_airfoilPost;
    sim.d_casePost = d_airfoilPost;
}

void airfoil_apply_boundary(Simulation &sim, int iter)
{
    auto *h_airfoil = static_cast<airfoilVar *>(sim.h_caseData);
    auto *d_airfoil = static_cast<airfoilVar *>(sim.d_caseData);

    const int NB = h_airfoil->NB;

    constexpr dim3 boundary_block(BLOCK_NODES);
    const size_t grid_block = (NB + BLOCK_NODES - 1) / BLOCK_NODES;

    dim3 boundary_grid(grid_block);

    apply_bc_airfoil<<<boundary_grid, boundary_block>>>(NB, INNER_NODE, *d_airfoil, sim.d_fMom, UXP_WALL, UYP_WALL,
                                                         UZP_WALL, D_WALL, iter);

    checkKernelExecution();
}

__global__ void apply_bc_airfoil(const int NB, const nodeType_t NODE_TYPE, const airfoilVar &airfoil,
                                  nodeVar dMom, const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                  const real D_WALL, const int iter)
{

    // boundary index
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= NB)
        return;

    // global index loaded from the boundary list
    const size_t idx = airfoil.boundaryList[i];

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

    if (nodeType >= NODE_TYPE && nodeType < (NODE_TYPE + NB))
    {
        const real delta = airfoil.delta_w[i];
        const real unit_nx = airfoil.unit_nx[i];
        const real unit_ny = airfoil.unit_ny[i];
        const uint32_t incomingMask = airfoil.incomingMask[i];
        const uint32_t outgoingMask = airfoil.outgoingMask[i];
        const real xw = XC + toReal(0.5) * D_WALL * unit_nx;
        const real yw = YC + toReal(0.5) * D_WALL * unit_ny;
        const real zw = toReal(z);

        curved_boundary_condition_rotated(unit_nx, unit_ny, delta, incomingMask, outgoingMask, x, y, z, xw, yw, zw, dMom,
                                          rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz,
                                          UX_PRIME, UY_PRIME, UZ_PRIME, iter);
    }

    // writing  moments into global memory (being done only for airfoil block)
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

__device__ void airfoil_boundary_moments(nodeType_t nodeType, airfoilVar &airfoil, nodeVar &dMom, real *pop,
                                          real &rho, real &ux, real &uy, real &uz,
                                          real &mxx, real &myy, real &mzz,
                                          real &mxy, real &mxz, real &myz)
{
    const int NB = airfoil.NB;
    if (nodeType >= INNER_NODE && nodeType < (INNER_NODE + NB))
    {
        const nodeType_t id = nodeType - INNER_NODE;
        const real unit_nx = airfoil.unit_nx[id];
        const real unit_ny = airfoil.unit_ny[id];
        const uint32_t incomingMask = airfoil.incomingMask[id];

        evaluate_incoming_moments_rotated(unit_nx, unit_ny, incomingMask, pop, rho, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (triangular && nodeType >= (BCFLUID_NODE + 0) && nodeType < (BCFLUID_NODE + 256))
    {
        const nodeType_t nodeTag = nodeType - BCFLUID_NODE;
        const uint32_t incomingMask = d_incomingMask_bcfluid[nodeTag];

        fluid_boundary_condition(nodeTag, incomingMask, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (!Z_PERIODIC && nodeType >= (BCSOLID_NODE + 0) && nodeType < (BCSOLID_NODE + 256))
    {
        const nodeType_t nodeTag = nodeType - BCSOLID_NODE;

        bcsolid_boundary_condition(nodeTag, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        boundary_condition(nodeType, dMom, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
}

//================================================ POST-PROCESS====================================================

void airfoil_post_streaming(Simulation &sim, int iter)
{
    airfoil_incoming_force_kernal(sim, iter);
}

void airfoil_post_collision(Simulation &sim, int iter)
{
    airfoil_outgoing_force_kernal(sim, iter);
}

void airfoil_post_process(Simulation &sim, int iter)
{
    if (iter >= STAT_START && iter <= STAT_END)
    {
        write_forces_mass(sim.h_fMom, iter);

        auto *h_airfoil = static_cast<airfoilVar *>(sim.h_caseData);
        auto *d_airfoil = static_cast<airfoilVar *>(sim.d_caseData);

        auto *h_airfoilPost = static_cast<airfoilPostProcess *>(sim.h_casePost);
        auto *d_airfoilPost = static_cast<airfoilPostProcess *>(sim.d_casePost);

        const int NB = h_airfoil->NB;

        constexpr dim3 boundary_block(BLOCK_NODES);
        const size_t grid_block = (NB + BLOCK_NODES - 1) / BLOCK_NODES;

        dim3 boundary_grid(grid_block);

        h_airfoilPost->n_avg++;
        compute_surface_pressure<<<grid_block, boundary_block>>>(sim.d_fMom, *d_airfoil, *d_airfoilPost,
                                                                 h_airfoilPost->n_avg);

        if (iter == STAT_END)
        {
            cudaMemcpy(h_airfoilPost->Cp_avg, d_airfoilPost->Cp_avg, NB * sizeof(real), cudaMemcpyDeviceToHost);
            write_pressure(*h_airfoil, *h_airfoilPost);
        }
    }
}

void airfoil_incoming_force_kernal(Simulation &sim, const int iter)
{
    if (iter >= STAT_START && iter <= STAT_END)
    {
        auto *h_airfoil = static_cast<airfoilVar *>(sim.h_caseData);
        auto *d_airfoil = static_cast<airfoilVar *>(sim.d_caseData);

        const int NB = h_airfoil->NB;
        const int NB_FLUID = h_airfoil->NB_FLUID;

        real zero = 0.0;
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFx, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFy, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFz, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_Totalm, &zero, sizeof(real)));

        const size_t FORCE_GRID = (NB + NB_FLUID + BLOCK_NODES - 1) / BLOCK_NODES;
        const dim3 force_grid(FORCE_GRID);

        compute_force_mass_kernel<<<force_grid, BLOCK_NODES>>>(sim.d_fMom,
                                                               d_airfoil->boundaryList,
                                                               d_airfoil->bcfluidList,
                                                               d_airfoil->incomingMask,
                                                               d_incomingMask_bcfluid,
                                                               d_airfoil->NB,
                                                               d_airfoil->NB_FLUID,
                                                               +1.0);
        checkKernelExecution();
    }
}

void airfoil_outgoing_force_kernal(Simulation &sim, const int iter)
{
    if (iter >= STAT_START && iter <= STAT_END)
    {
        auto *h_airfoil = static_cast<airfoilVar *>(sim.h_caseData);
        auto *d_airfoil = static_cast<airfoilVar *>(sim.d_caseData);

        const int NB = h_airfoil->NB;
        const int NB_FLUID = h_airfoil->NB_FLUID;

        const size_t FORCE_GRID = (NB + NB_FLUID + BLOCK_NODES - 1) / BLOCK_NODES;
        const dim3 force_grid(FORCE_GRID);

        compute_force_mass_kernel<<<force_grid, BLOCK_NODES>>>(sim.d_fMom,
                                                               d_airfoil->boundaryList,
                                                               d_airfoil->bcfluidList,
                                                               d_airfoil->outgoingMask,
                                                               d_outgoingMask_bcfluid,
                                                               d_airfoil->NB,
                                                               d_airfoil->NB_FLUID,
                                                               -1.0);
        checkKernelExecution();
    }
}

__global__ void compute_surface_pressure(const nodeVar &dMom, const airfoilVar &d_airfoil,
                                         const airfoilPostProcess &d_airfoilPost,
                                         const int n_avg)
{
    const int nb = d_airfoil.NB;
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nb)
        return;

    const size_t idx = d_airfoil.boundaryList[i];
    unsigned int x, y, z;
    GlobalIndexToXYZ(idx, x, y, z);

    const real unit_nx = d_airfoil.unit_nx[i];
    const real unit_ny = d_airfoil.unit_ny[i];

    // wall point location (airfoil)
    const real rmax = toReal(0.5) * D_WALL;
    const real xw = XC + rmax * unit_nx;
    const real yw = YC + rmax * unit_ny;

    // First reference fluid point and pressure calculation
    const real x1 = xw + delx * unit_nx;
    const real y1 = yw + delx * unit_ny;
    const real rho1 = RHO_0 + bilinear_interpolation(x1, y1, z, dMom.rho);

    // Second reference fluid point and pressure calculation
    const real x2 = xw + toReal(2.0) * delx * unit_nx;
    const real y2 = yw + toReal(2.0) * delx * unit_ny;
    const real rho2 = RHO_0 + bilinear_interpolation(x2, y2, z, dMom.rho);

    // Third reference fluid point and pressure calculation
    const real x3 = xw + toReal(3.0) * delx * unit_nx;
    const real y3 = yw + toReal(3.0) * delx * unit_ny;
    const real rho3 = RHO_0 + bilinear_interpolation(x3, y3, z, dMom.rho);

    // surface pressure extrapolation
    const real rho_s = surface_pressure_extrapolation(xw, yw, x1, y1, x2, y2, x3, y3, rho1, rho2, rho3);

    real rho_avg = d_airfoilPost.Cp_avg[i];

    rho_avg += (rho_s - rho_avg) / toReal(n_avg);

    d_airfoilPost.Cp_avg[i] = rho_avg * cs2;
}
