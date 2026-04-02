#include "cylinder_kernels.cuh"

#ifdef CYLINDER

void cylinder_initialize(Simulation &sim)
{
    // allocate struct
    cylinderVar *h_cylinder = new cylinderVar;
    cylinderVar *d_cylinder = new cylinderVar;

    cylinderPostProcess *h_cylinderPost = new cylinderPostProcess;
    cylinderPostProcess *d_cylinderPost = new cylinderPostProcess;

    // compute geometry + counts (NB, etc.)
    triangular ? initialize_cylinder_nodeType_triangular(sim.h_fMom, h_cylinder->inner)
               : initialize_cylinder_nodeType_staircase(sim.h_fMom, h_cylinder->inner);

    // allocate memory using computed sizes
    allocatecylinderMemory(h_cylinder->inner, d_cylinder->inner, *h_cylinderPost, *d_cylinderPost);

    buildBoundaryList_updateBoundaryNodeType(sim.h_fMom, h_cylinder->inner,
                                             NODE_INNER, NODE_BCFLUID_INNER, NODE_BCSOLID_INNER);

    // modify nodeType for boundary nodes
    assignBoundaryIndices(sim.h_fMom, h_cylinder->inner, NODE_INNER);

    compute_unit_vectors_boundary_nodes(sim.h_fMom, h_cylinder->inner, D_WALL);

    find_incomings_outgoings(sim.h_fMom, h_cylinder->inner);

    if constexpr (triangular)
    {
        setup_bcfluid_masks(sim.h_fMom, h_cylinder->inner);
        setup_bcsolid_masks(sim.h_fMom, h_cylinder->inner);
        cylinder_host_device_constants();
    }

    // copy data to device arrays
    copyHostToDevice(d_cylinder->inner, h_cylinder->inner);

    // store in simulation
    sim.h_caseData = h_cylinder;
    sim.d_caseData = d_cylinder;

    sim.h_casePost = h_cylinderPost;
    sim.d_casePost = d_cylinderPost;

    write_geometry_files(sim.h_fMom);
}

void cylinder_apply_boundary(Simulation &sim, int iter)
{
    auto *h_cylinder = static_cast<cylinderVar *>(sim.h_caseData);
    auto *d_cylinder = static_cast<cylinderVar *>(sim.d_caseData);

    auto launch_bc = [&](boundaryVar &h_b, boundaryVar &d_b, real ux, real uy, real uz, real D, int iter)
    {
        int NB = h_b.NB;
        const dim3 boundary_block(BLOCK_NODES);
        const dim3 boundary_grid((NB + BLOCK_NODES - 1) / BLOCK_NODES);
        apply_bc_cylinder<<<boundary_grid, boundary_block>>>(NB, d_b, sim.d_fMom, ux, uy, uz, D, iter);
        checkKernelExecution();
    };

    launch_bc(h_cylinder->inner, d_cylinder->inner, UXP_INNER, UYP_INNER, UZP_INNER, D_WALL, iter);
}

__global__ void apply_bc_cylinder(const int NB, const boundaryVar &cylinder, nodeVar dMom,
                                  const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
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

    const nodeType_t nodeType_packed = dMom.nodeType[idx];
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

    const nodeType_t nodeType = getType(nodeType_packed);
    const nodeType_t tag = getIndex(nodeType_packed);
    if (tag >= cylinder.NB)
    {
        printf("ERROR: invalid tag %u at idx %zu\n", tag, idx);
    }

    const real delta = cylinder.delta_w[tag];
    const real unit_nx = cylinder.unit_nx[tag];
    const real unit_ny = cylinder.unit_ny[tag];
    const uint32_t incomingMask = cylinder.incomingMask[tag];
    const uint32_t outgoingMask = cylinder.outgoingMask[tag];

    const real xw = XC + toReal(0.5) * D_WALL * unit_nx;
    const real yw = YC + toReal(0.5) * D_WALL * unit_ny;
    const real zw = toReal(z);

    curved_boundary_condition_rotated(nodeType, unit_nx, unit_ny, delta, incomingMask, outgoingMask, x, y, z, xw, yw, zw, dMom,
                                      rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz,
                                      UX_PRIME, UY_PRIME, UZ_PRIME, iter);

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

//================================================ POST-PROCESS====================================================
void cylinder_write_output(Simulation &sim, int iter)
{
    write_vti_3d(sim.h_fMom, iter);
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
        write_forces_mass(sim.h_fMom, iter);

        auto *h_cylinder = static_cast<cylinderVar *>(sim.h_caseData);
        auto *d_cylinder = static_cast<cylinderVar *>(sim.d_caseData);

        auto *h_cylinderPost = static_cast<cylinderPostProcess *>(sim.h_casePost);
        auto *d_cylinderPost = static_cast<cylinderPostProcess *>(sim.d_casePost);

        const int NB = h_cylinder->inner.NB;

        constexpr dim3 boundary_block(BLOCK_NODES);
        const size_t grid_block = (NB + BLOCK_NODES - 1) / BLOCK_NODES;

        dim3 boundary_grid(grid_block);

        h_cylinderPost->n_avg++;
        compute_surface_pressure<<<grid_block, boundary_block>>>(sim.d_fMom, NB, d_cylinder->inner.boundaryList,
                                                                 d_cylinder->inner.unit_nx, d_cylinder->inner.unit_ny,
                                                                 d_cylinderPost->ps_avg, h_cylinderPost->n_avg);

        if (iter == STAT_END)
        {
            cudaMemcpy(h_cylinderPost->ps_avg, d_cylinderPost->ps_avg, NB * sizeof(real), cudaMemcpyDeviceToHost);
            write_pressure(*h_cylinder, *h_cylinderPost);
        }
    }
}

void cylinder_incoming_force_kernal(Simulation &sim, const int iter)
{
    if (iter >= STAT_START && iter <= STAT_END)
    {
        auto *h_cylinder = static_cast<cylinderVar *>(sim.h_caseData);
        auto *d_cylinder = static_cast<cylinderVar *>(sim.d_caseData);

        const int NB = h_cylinder->inner.NB;

        const int NB_FLUID = h_cylinder->inner.NB_FLUID;

        real zero = 0.0;
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFx, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFy, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_TotalFz, &zero, sizeof(real)));
        checkCudaErrors(cudaMemcpyToSymbol(d_Totalm, &zero, sizeof(real)));

        const size_t FORCE_GRID = (NB + NB_FLUID + BLOCK_NODES - 1) / BLOCK_NODES;
        const dim3 force_grid(FORCE_GRID);

        compute_force_mass_kernel<<<force_grid, BLOCK_NODES>>>(sim.d_fMom,
                                                               d_cylinder->inner.boundaryList,
                                                               d_cylinder->inner.bcfluidList,
                                                               d_cylinder->inner.incomingMask,
                                                               d_cylinder->inner.NB,
                                                               d_cylinder->inner.NB_FLUID,
                                                               MaskType::INCOMING,
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

        const int NB = h_cylinder->inner.NB;
        const int NB_FLUID = h_cylinder->inner.NB_FLUID;

        const size_t FORCE_GRID = (NB + NB_FLUID + BLOCK_NODES - 1) / BLOCK_NODES;
        const dim3 force_grid(FORCE_GRID);

        compute_force_mass_kernel<<<force_grid, BLOCK_NODES>>>(sim.d_fMom,
                                                               d_cylinder->inner.boundaryList,
                                                               d_cylinder->inner.bcfluidList,
                                                               d_cylinder->inner.outgoingMask,
                                                               d_cylinder->inner.NB,
                                                               d_cylinder->inner.NB_FLUID,
                                                               MaskType::OUTGOING,
                                                               -1.0);
        checkKernelExecution();
    }
}

__global__ void compute_surface_pressure(const nodeVar &dMom, const int nb,
                                         size_t *boundaryList, real *d_unit_nx, real *d_unit_ny,
                                         real *d_ps_avg, const int n_avg)
{
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nb)
        return;

    const size_t idx = boundaryList[i];
    unsigned int x, y, z;
    GlobalIndexToXYZ(idx, x, y, z);

    const real unit_nx = d_unit_nx[i];
    const real unit_ny = d_unit_ny[i];

    // wall point location (cylinder)
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
    const real ps = rho_s * cs2;

    real ps_avg = d_ps_avg[i];

    // incremental average
    ps_avg += (ps - ps_avg) / toReal(n_avg);

    d_ps_avg[i] = ps_avg;
}

#endif