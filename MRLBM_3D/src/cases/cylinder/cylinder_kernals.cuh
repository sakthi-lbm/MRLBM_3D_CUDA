#pragma once

#include "cylinder_helpers.cuh"

void cylinder_initialize(Simulation &sim);
void cylinder_apply_boundary(Simulation &sim, int iter);
void cylinder_free(Simulation &sim);
void setup_cylinder_case(Case &case_module);

__device__ void cylinder_boundary_moments(nodeType_t nodeType, cylinderVar &cylinder, nodeVar &dMom, real *pop,
                                          real &rho, real &ux, real &uy, real &uz,
                                          real &mxx, real &myy, real &mzz,
                                          real &mxy, real &mxz, real &myz);

__global__ void apply_bc_cylinder(const int NB, const nodeType_t NODE_TYPE, const cylinderVar &cylinder,
                                  nodeVar dMom, const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                  const real D_WALL, const int iter);

inline void allocateCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    const int NB = h_cylinder.NB;
    const int NB_FLUID = h_cylinder.NB_FLUID;
    const int NB_SOLID = h_cylinder.NB_SOLID;

    if (NB <= 0)
    {
        std::cerr << "Error: NB not initialized!\n";
        exit(EXIT_FAILURE);
    }

    // ================= HOST =================
    checkCudaErrors(cudaMallocHost(&h_cylinder.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMallocHost(&h_cylinder.bcfluidList, NB_FLUID * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.bcsolidList, NB_SOLID * sizeof(size_t)));

    checkCudaErrors(cudaMallocHost(&h_cylinder.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.delta_w, NB * sizeof(real)));

    // ================= DEVICE =================
    checkCudaErrors(cudaMalloc(&d_cylinder.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMalloc(&d_cylinder.bcfluidList, NB_FLUID * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.bcsolidList, NB_SOLID * sizeof(size_t)));

    checkCudaErrors(cudaMalloc(&d_cylinder.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_cylinder.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_cylinder.delta_w, NB * sizeof(real)));
}

inline void freeCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    // Free-ing host memory
    checkCudaErrors(cudaFreeHost(h_cylinder.boundaryList));
    checkCudaErrors(cudaFreeHost(h_cylinder.incomingMask));
    checkCudaErrors(cudaFreeHost(h_cylinder.outgoingMask));

    checkCudaErrors(cudaFreeHost(h_cylinder.bcfluidList));
    checkCudaErrors(cudaFreeHost(h_cylinder.bcsolidList));

    // Free-ing device memory
    checkCudaErrors(cudaFree(d_cylinder.boundaryList));
    checkCudaErrors(cudaFree(d_cylinder.incomingMask));
    checkCudaErrors(cudaFree(d_cylinder.outgoingMask));

    checkCudaErrors(cudaFree(d_cylinder.bcfluidList));
    checkCudaErrors(cudaFree(d_cylinder.bcsolidList));
}

inline void copyHostToDevice(cylinderVar &d_cylinder, cylinderVar &h_cylinder)
{
    const int NB = h_cylinder.NB;
    const int NB_FLUID = h_cylinder.NB_FLUID;
    const int NB_SOLID = h_cylinder.NB_SOLID;

    d_cylinder.NB = NB;
    d_cylinder.NB_FLUID = NB_FLUID;
    d_cylinder.NB_SOLID = NB_SOLID;

    // Copy arrays (host → device)
    checkCudaErrors(cudaMemcpy(d_cylinder.boundaryList, h_cylinder.boundaryList, NB * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.incomingMask, h_cylinder.incomingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.outgoingMask, h_cylinder.outgoingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.bcfluidList, h_cylinder.bcfluidList, NB_FLUID * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.bcsolidList, h_cylinder.bcsolidList, NB_SOLID * sizeof(size_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_cylinder.unit_nx, h_cylinder.unit_nx, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.unit_ny, h_cylinder.unit_ny, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.delta_w, h_cylinder.delta_w, NB * sizeof(real), cudaMemcpyHostToDevice));
}

inline void cylinder_host_device_constants()
{
    cudaMemcpyToSymbol(d_incomingMask_bcfluid, h_incomingMask_bcfluid, MAX_NODE_TAG * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_outgoingMask_bcfluid, h_outgoingMask_bcfluid, MAX_NODE_TAG * sizeof(uint32_t));

    cudaMemcpyToSymbol(d_incomingMask_bcsolid, h_incomingMask_bcsolid, MAX_NODE_TAG * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_outgoingMask_bcsolid, h_outgoingMask_bcsolid, MAX_NODE_TAG * sizeof(uint32_t));
}