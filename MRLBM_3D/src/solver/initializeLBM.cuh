#ifndef INITIALIZELBM_H
#define INITIALIZELBM_H

#include "cylinderLBM.cuh"

void initialize_domain(nodeVar &dMom, nodeVar &hMom, haloData &gHalo, cylinderVar &h_cylinder, cylinderVar &d_cylinder);
__global__ void gpu_initialize_Moments_nodeType_GhostInterface(nodeVar dMom, haloData gHalo);

inline void allocateCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    // Host Memory
    checkCudaErrors(cudaMallocHost(&h_cylinder.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMallocHost(&h_cylinder.bcfluidList, NB_FLUID * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.bcsolidList, NB_SOLID * sizeof(size_t)));

    // Device Memory
    checkCudaErrors(cudaMalloc(&d_cylinder.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMalloc(&d_cylinder.bcfluidList, NB_FLUID * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.bcsolidList, NB_SOLID * sizeof(size_t)));
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
    checkCudaErrors(cudaMemcpy(d_cylinder.boundaryList, h_cylinder.boundaryList, NB * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.incomingMask, h_cylinder.incomingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.outgoingMask, h_cylinder.outgoingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_cylinder.bcfluidList, h_cylinder.bcfluidList, NB_FLUID * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.bcsolidList, h_cylinder.bcsolidList, NB_SOLID * sizeof(size_t), cudaMemcpyHostToDevice));
}

#endif // INITIALIZELBM_H