#include "cylinder_kernals.cuh"

void cylinder_initialize(Simulation &sim)
{
    // allocate struct
    cylinderVar *h_cylinder = new cylinderVar;
    cylinderVar *d_cylinder = new cylinderVar;

    // compute geometry + counts (NB, etc.)
    triangular ? initialize_cylinder_nodeType_triangular(sim.h_fMom, *h_cylinder)
               : initialize_cylinder_nodeType_staircase(sim.h_fMom, *h_cylinder);

    write_geometry_files(sim.h_fMom);

    // allocate memory using computed sizes
    allocateCylinderMemory(*h_cylinder, *d_cylinder);

    buildBoundaryList_updateBoundaryNodeType(sim.h_fMom, *h_cylinder);
    compute_unit_vectors_boundary_nodes(sim.h_fMom, *h_cylinder, D_WALL);
    find_incomings_outgoings(sim.h_fMom, *h_cylinder);
    setup_bcfluid_masks(sim.h_fMom, *h_cylinder);
    setup_bcsolid_masks(sim.h_fMom, *h_cylinder);

    // copy data to device arrays
    copyHostToDevice(*d_cylinder, *h_cylinder);
    cylinder_host_device_constants();

    // store in simulation
    sim.h_caseData = h_cylinder;
    sim.d_caseData = d_cylinder;
}

void cylinder_apply_boundary(Simulation &sim, int iter)
{
    auto *h_cylinder = static_cast<cylinderVar *>(sim.h_caseData);
    auto *d_cylinder = static_cast<cylinderVar *>(sim.d_caseData);

    const int NB = h_cylinder->NB;

    constexpr dim3 boundary_block(BLOCK_NODES);
    const size_t grid_block = (NB + BLOCK_NODES - 1) / BLOCK_NODES;

    dim3 boundary_grid(grid_block);

    apply_bc_cylinder<<<boundary_grid, boundary_block>>>(NB, INNER_NODE, *d_cylinder, sim.d_fMom, UXP_CYLINDER, UYP_CYLINDER,
                                                         UZP_CYLINDER, D_WALL, iter);

    checkKernelExecution();
}

void cylinder_post_streaming(Simulation &sim, int iter)
{
    cylinder_incoming_force_kernal(sim, iter);
}

void cylinder_post_collision(Simulation &sim, int iter)
{
    cylinder_outgoing_force_kernal(sim, iter);
}

void cylinder_post_process(Simulation &sim, int iter)
{
    if (iter >= STAT_START && iter <= STAT_END)
    {
        write_statistics(sim.h_fMom, iter);
    }
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

    if (nodeType >= NODE_TYPE && nodeType < (NODE_TYPE + NB))
    {
        const real delta = cylinder.delta_w[i];
        const real unit_nx = cylinder.unit_nx[i];
        const real unit_ny = cylinder.unit_ny[i];
        const real xw = XC + toReal(0.5) * D_WALL * unit_nx;
        const real yw = YC + toReal(0.5) * D_WALL * unit_ny;
        const real zw = toReal(z);

        curved_boundary_condition_rotated(unit_nx, unit_ny, delta, x, y, z, xw, yw, zw, cylinder, nodeType, dMom,
                                          rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz,
                                          UX_PRIME, UY_PRIME, UZ_PRIME, NODE_TYPE, iter);
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

__device__ void cylinder_boundary_moments(nodeType_t nodeType, cylinderVar &cylinder, nodeVar &dMom, real *pop,
                                          real &rho, real &ux, real &uy, real &uz,
                                          real &mxx, real &myy, real &mzz,
                                          real &mxy, real &mxz, real &myz)
{
    const int NB = cylinder.NB;
    if (nodeType >= INNER_NODE && nodeType < (INNER_NODE + NB))
    {
        const nodeType_t id = nodeType - INNER_NODE;
        const real unit_nx = cylinder.unit_nx[id];
        const real unit_ny = cylinder.unit_ny[id];
        const uint32_t incomingMask = cylinder.incomingMask[id];

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

void cylinder_incoming_force_kernal(Simulation &sim, const int iter)
{
    if (iter >= STAT_START && iter <= STAT_END)
    {
        auto *h_cylinder = static_cast<cylinderVar *>(sim.h_caseData);
        auto *d_cylinder = static_cast<cylinderVar *>(sim.d_caseData);

        const int NB = h_cylinder->NB;
        const int NB_FLUID = h_cylinder->NB_FLUID;

        real zero = 0.0;
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFx, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFy, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFz, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_Totalm, &zero, sizeof(real)));

        const size_t FORCE_GRID = (NB + NB_FLUID + BLOCK_NODES - 1) / BLOCK_NODES;
        const dim3 force_grid(FORCE_GRID);

        compute_force_mass_kernel<<<force_grid, BLOCK_NODES>>>(sim.d_fMom,
                                                               d_cylinder->boundaryList,
                                                               d_cylinder->bcfluidList,
                                                               d_cylinder->incomingMask,
                                                               d_incomingMask_bcfluid,
                                                               d_cylinder->NB,
                                                               d_cylinder->NB_FLUID,
                                                               +1.0);
        checkKernelExecution();
    }
}

void cylinder_outgoing_force_kernal(Simulation &sim, const int iter)
{
    if (iter >= STAT_START && iter <= STAT_END)
    {
        auto *h_cylinder = static_cast<cylinderVar *>(sim.h_caseData);
        auto *d_cylinder = static_cast<cylinderVar *>(sim.d_caseData);

        const int NB = h_cylinder->NB;
        const int NB_FLUID = h_cylinder->NB_FLUID;

        const size_t FORCE_GRID = (NB + NB_FLUID + BLOCK_NODES - 1) / BLOCK_NODES;
        const dim3 force_grid(FORCE_GRID);

        compute_force_mass_kernel<<<force_grid, BLOCK_NODES>>>(sim.d_fMom,
                                                               d_cylinder->boundaryList,
                                                               d_cylinder->bcfluidList,
                                                               d_cylinder->outgoingMask,
                                                               d_outgoingMask_bcfluid,
                                                               d_cylinder->NB,
                                                               d_cylinder->NB_FLUID,
                                                               -1.0);
        checkKernelExecution();
    }
}

void cylinder_free(Simulation &sim)
{
    auto *h_cyl = static_cast<cylinderVar *>(sim.h_caseData);
    auto *d_cyl = static_cast<cylinderVar *>(sim.d_caseData);

    if (!h_cyl || !d_cyl)
        return;

    // Free host memory
    cudaFreeHost(h_cyl->boundaryList);
    cudaFreeHost(h_cyl->incomingMask);
    cudaFreeHost(h_cyl->outgoingMask);

    cudaFreeHost(h_cyl->bcfluidList);
    cudaFreeHost(h_cyl->bcsolidList);

    cudaFreeHost(h_cyl->unit_nx);
    cudaFreeHost(h_cyl->unit_ny);
    cudaFreeHost(h_cyl->delta_w);

    // Free device memory
    cudaFree(d_cyl->boundaryList);
    cudaFree(d_cyl->incomingMask);
    cudaFree(d_cyl->outgoingMask);

    cudaFree(d_cyl->bcfluidList);
    cudaFree(d_cyl->bcsolidList);

    cudaFree(d_cyl->unit_nx);
    cudaFree(d_cyl->unit_ny);
    cudaFree(d_cyl->delta_w);

    // Delete structs
    delete h_cyl;
    delete d_cyl;

    sim.h_caseData = nullptr;
    sim.d_caseData = nullptr;
}