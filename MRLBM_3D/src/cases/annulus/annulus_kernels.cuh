#pragma once

#include "annulus_helpers.cuh"

void annulus_initialize(Simulation &sim);
void annulus_apply_boundary(Simulation &sim, int iter);
void setup_annulus_case(Case &case_module);

__global__ void apply_bc_annulus(const int NB, const boundaryVar &annulus, nodeVar dMom,
                                 const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                 const real D_WALL, const int iter);

//=================================================================================================================

__device__ __forceinline__ void annulus_boundary_moments(const nodeType_t nodeType_packed,
                                                         const boundaryVar &inner, const boundaryVar &outer,
                                                         nodeVar &dMom, real *pop,
                                                         real &rho, real &ux, real &uy, real &uz,
                                                         real &mxx, real &myy, real &mzz,
                                                         real &mxy, real &mxz, real &myz)
{
    const nodeType_t nodeType = getType(nodeType_packed);
    const nodeType_t tag = getIndex(nodeType_packed);

    if (nodeType == NODE_INNER)
    {
        const real unit_nx = inner.unit_nx[tag];
        const real unit_ny = inner.unit_ny[tag];
        const uint32_t incomingMask = inner.incomingMask[tag];

        evaluate_incoming_moments_rotated(unit_nx, unit_ny, incomingMask, pop, rho, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (nodeType == NODE_OUTER)
    {
        const real unit_nx = outer.unit_nx[tag];
        const real unit_ny = outer.unit_ny[tag];
        const uint32_t incomingMask = outer.incomingMask[tag];

        evaluate_incoming_moments_rotated(unit_nx, unit_ny, incomingMask, pop, rho, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (triangular && (nodeType == NODE_BCFLUID_INNER || nodeType == NODE_BCFLUID_OUTER))
    {
        // printf("%d %d \n", toInt(nodeType), toInt(tag));
        const uint32_t incomingMask = d_incomingMask_bcfluid[tag];

        fluid_boundary_condition(tag, incomingMask, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (!Z_PERIODIC && (nodeType == NODE_BCSOLID_INNER || nodeType == NODE_BCSOLID_OUTER))
    {
        bcsolid_boundary_condition(tag, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        // boundary_condition(nodeType, dMom, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
}

inline void allocateAnnulusMemory(boundaryVar &h_annulus, boundaryVar &d_annulus)
{
    const int NB = h_annulus.NB;

    if (NB <= 0)
    {
        std::cerr << "Error: NB not initialized!\n";
        exit(EXIT_FAILURE);
    }

    // ================= HOST =================
    checkCudaErrors(cudaMallocHost(&h_annulus.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_annulus.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMallocHost(&h_annulus.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMallocHost(&h_annulus.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_annulus.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_annulus.delta_w, NB * sizeof(real)));

    // ================= DEVICE =================
    checkCudaErrors(cudaMalloc(&d_annulus.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_annulus.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&d_annulus.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMalloc(&d_annulus.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_annulus.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_annulus.delta_w, NB * sizeof(real)));

    if constexpr (triangular)
    {
        const int NB_FLUID = h_annulus.NB_FLUID;
        const int NB_SOLID = h_annulus.NB_SOLID;

        checkCudaErrors(cudaMallocHost(&h_annulus.bcfluidList, NB_FLUID * sizeof(size_t)));
        checkCudaErrors(cudaMallocHost(&h_annulus.bcsolidList, NB_SOLID * sizeof(size_t)));

        checkCudaErrors(cudaMalloc(&d_annulus.bcfluidList, NB_FLUID * sizeof(size_t)));
        checkCudaErrors(cudaMalloc(&d_annulus.bcsolidList, NB_SOLID * sizeof(size_t)));
    }
}

inline void free_boundary(boundaryVar &h_b, boundaryVar &d_b)
{
    // host
    cudaFreeHost(h_b.boundaryList);
    cudaFreeHost(h_b.incomingMask);
    cudaFreeHost(h_b.outgoingMask);

    cudaFreeHost(h_b.bcfluidList);
    cudaFreeHost(h_b.bcsolidList);

    cudaFreeHost(h_b.unit_nx);
    cudaFreeHost(h_b.unit_ny);
    cudaFreeHost(h_b.delta_w);

    // device
    cudaFree(d_b.boundaryList);
    cudaFree(d_b.incomingMask);
    cudaFree(d_b.outgoingMask);

    cudaFree(d_b.bcfluidList);
    cudaFree(d_b.bcsolidList);

    cudaFree(d_b.unit_nx);
    cudaFree(d_b.unit_ny);
    cudaFree(d_b.delta_w);
}

inline void annulus_free(Simulation &sim)
{

    auto *h_annulus = static_cast<annulusVar *>(sim.h_caseData);
    auto *d_annulus = static_cast<annulusVar *>(sim.d_caseData);

    if (!h_annulus || !d_annulus)
        return;

    free_boundary(h_annulus->inner, d_annulus->inner);
    free_boundary(h_annulus->outer, d_annulus->outer);

    // delete structs
    delete h_annulus;
    delete d_annulus;

    sim.h_caseData = nullptr;
    sim.d_caseData = nullptr;
}

inline void copyHostToDevice(boundaryVar &d_annulus, boundaryVar &h_annulus)
{
    const int NB = h_annulus.NB;
    d_annulus.NB = NB;

    // Copy arrays (host → device)
    checkCudaErrors(cudaMemcpy(d_annulus.boundaryList, h_annulus.boundaryList, NB * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_annulus.incomingMask, h_annulus.incomingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_annulus.outgoingMask, h_annulus.outgoingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_annulus.unit_nx, h_annulus.unit_nx, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_annulus.unit_ny, h_annulus.unit_ny, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_annulus.delta_w, h_annulus.delta_w, NB * sizeof(real), cudaMemcpyHostToDevice));

    if (triangular)
    {
        const int NB_FLUID = h_annulus.NB_FLUID;
        const int NB_SOLID = h_annulus.NB_SOLID;
        d_annulus.NB_FLUID = NB_FLUID;
        d_annulus.NB_SOLID = NB_SOLID;

        checkCudaErrors(cudaMemcpy(d_annulus.bcfluidList, h_annulus.bcfluidList, NB_FLUID * sizeof(size_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_annulus.bcsolidList, h_annulus.bcsolidList, NB_SOLID * sizeof(size_t), cudaMemcpyHostToDevice));
    }
}

inline void annulus_host_device_constants()
{
    cudaMemcpyToSymbol(d_incomingMask_bcfluid, h_incomingMask_bcfluid, MAX_NODE_TAG * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_outgoingMask_bcfluid, h_outgoingMask_bcfluid, MAX_NODE_TAG * sizeof(uint32_t));

    cudaMemcpyToSymbol(d_incomingMask_bcsolid, h_incomingMask_bcsolid, MAX_NODE_TAG * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_outgoingMask_bcsolid, h_outgoingMask_bcsolid, MAX_NODE_TAG * sizeof(uint32_t));
}
