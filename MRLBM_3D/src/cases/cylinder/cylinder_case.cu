#include "cylinder_kernals.cuh"

__constant__ uint32_t d_incomingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_incomingMask_bcsolid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcsolid[MAX_NODE_TAG];

void setup_cylinder_case(Case &case_module)
{
    case_module.initialize = cylinder_initialize;
    case_module.apply_boundary = cylinder_apply_boundary;

    case_module.compute_forces = nullptr;    // optional
    case_module.post_process_step = nullptr; // optional
    case_module.finalize = nullptr;

    case_module.free_case = cylinder_free;
}

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