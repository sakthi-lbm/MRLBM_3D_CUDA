#include "annulus_kernels.cuh"

void annulus_initialize(Simulation &sim)
{
    // allocate struct
    annulusVar *h_annulus = new annulusVar;
    annulusVar *d_annulus = new annulusVar;

    // compute geometry + counts (NB, etc.)
    triangular ? initialize_annulus_nodeType_triangular(sim.h_fMom, h_annulus->inner, h_annulus->outer)
               : initialize_annulus_nodeType_staircase(sim.h_fMom, h_annulus->inner, h_annulus->outer);

    write_geometry_files(sim.h_fMom);

    // allocate memory using computed sizes
    allocateAnnulusMemory(h_annulus->inner, d_annulus->inner);
    allocateAnnulusMemory(h_annulus->outer, d_annulus->outer);

    buildBoundaryList_updateBoundaryNodeType(sim.h_fMom, h_annulus->inner,
                                             INNER_NODE, BCFLUID_NODE_INNER, BCSOLID_NODE_INNER);
    buildBoundaryList_updateBoundaryNodeType(sim.h_fMom, h_annulus->outer,
                                             OUTER_NODE, BCFLUID_NODE_OUTER, BCSOLID_NODE_OUTER);

    // modify nodeType for boundary nodes
    assignBoundaryIndices(sim.h_fMom, h_annulus->inner, INNER_NODE);
    assignBoundaryIndices(sim.h_fMom, h_annulus->outer, OUTER_NODE);

    compute_unit_vectors_boundary_nodes(sim.h_fMom, h_annulus->inner, D_IN);
    compute_unit_vectors_boundary_nodes(sim.h_fMom, h_annulus->outer, D_OUT);

    find_incomings_outgoings(sim.h_fMom, h_annulus->inner);
    find_incomings_outgoings(sim.h_fMom, h_annulus->outer);

    setup_bcfluid_masks(sim.h_fMom, h_annulus->inner, BCFLUID_NODE_INNER);
    setup_bcfluid_masks(sim.h_fMom, h_annulus->outer, BCFLUID_NODE_OUTER);

    setup_bcsolid_masks(sim.h_fMom, h_annulus->inner, BCSOLID_NODE_INNER);
    setup_bcsolid_masks(sim.h_fMom, h_annulus->outer, BCSOLID_NODE_OUTER);

    // copy data to device arrays
    copyHostToDevice(d_annulus->inner, h_annulus->inner);
    copyHostToDevice(d_annulus->outer, h_annulus->outer);

    annulus_host_device_constants();

    // store in simulation
    sim.h_caseData = h_annulus;
    sim.d_caseData = d_annulus;
}

void annulus_apply_boundary(Simulation &sim, int iter)
{
    auto *h_annulus = static_cast<annulusVar *>(sim.h_caseData);
    auto *d_annulus = static_cast<annulusVar *>(sim.d_caseData);

    auto launch_bc = [&](boundaryVar &h_b, boundaryVar &d_b, int node_type, real ux, real uy, real uz, real D)
    {
        int NB = h_b.NB;
        dim3 grid((NB + BLOCK_NODES - 1) / BLOCK_NODES);
        apply_bc_annulus<<<grid, block>>>(NB, node_type, d_b, sim.d_fMom, ux, uy, uz, D, iter);
        checkKernelExecution();
    };

    launch_bc(h_annulus->inner, d_annulus->inner, INNER_NODE, UXP_INNER, UYP_INNER, UZP_INNER, D_IN);
    launch_bc(h_annulus->outer, d_annulus->outer, OUTER_NODE, UXP_OUTER, UYP_OUTER, UZP_OUTER, D_OUT);
}

__global__ void apply_bc_annulus(const int NB, const nodeType_t NODE_TYPE, const boundaryVar &annulus,
                                 nodeVar dMom, const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                 const real D_WALL, const int iter)
{

    // boundary index
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= NB)
        return;

    // global index loaded from the boundary list
    const size_t idx = annulus.boundaryList[i];

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
        const real delta = annulus.delta_w[i];
        const real unit_nx = annulus.unit_nx[i];
        const real unit_ny = annulus.unit_ny[i];
        const uint32_t incomingMask = annulus.incomingMask[i];
        const uint32_t outgoingMask = annulus.outgoingMask[i];
        const real xw = XC + toReal(0.5) * D_WALL * unit_nx;
        const real yw = YC + toReal(0.5) * D_WALL * unit_ny;
        const real zw = toReal(z);

        curved_boundary_condition_rotated(unit_nx, unit_ny, delta, incomingMask, outgoingMask, x, y, z, xw, yw, zw, dMom,
                                          rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz,
                                          UX_PRIME, UY_PRIME, UZ_PRIME, iter);
    }

    // writing  moments into global memory (being done only for annulus block)
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
