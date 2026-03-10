#ifndef INITIALIZELBM_H
#define INITIALIZELBM_H

#include "cylinderLBM.cuh"

void initialize_domain(nodeVar &dMom, nodeVar &hMom, haloData &gHalo, cylinderVar &h_cylinder, cylinderVar &d_cylinder);
__global__ void gpu_initialize_Moments_nodeType_GhostInterface(nodeVar dMom, haloData gHalo);

inline void allocateCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder, const size_t nb)
{
    // Host Memory
    checkCudaErrors(cudaMallocHost(&h_cylinder.boundaryList, nb * sizeof(size_t)));
    cudaMallocHost(&h_cylinder.incomingMask, nb * sizeof(uint32_t));
    cudaMallocHost(&h_cylinder.outgoingMask, nb * sizeof(uint32_t));

    // Device Memory
    checkCudaErrors(cudaMalloc(&d_cylinder.boundaryList, nb * sizeof(size_t)));
    cudaMalloc(&d_cylinder.incomingMask, nb * sizeof(uint32_t));
    cudaMalloc(&d_cylinder.outgoingMask, nb * sizeof(uint32_t));
}

inline void freeCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    // Free-ing host memory
    cudaFreeHost(h_cylinder.boundaryList);
    cudaFreeHost(h_cylinder.incomingMask);
    cudaFreeHost(h_cylinder.outgoingMask);

    // Free-ing device memory
    cudaFree(d_cylinder.boundaryList);
    cudaFree(d_cylinder.incomingMask);
    cudaFree(d_cylinder.outgoingMask);
}

inline void copyHostToDevice(cylinderVar &d_cylinder, cylinderVar &h_cylinder, const size_t nb)
{
    checkCudaErrors(cudaMemcpy(d_cylinder.boundaryList, h_cylinder.boundaryList, nb * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.incomingMask, h_cylinder.incomingMask, nb * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.outgoingMask, h_cylinder.outgoingMask, nb * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

#endif // INITIALIZELBM_H