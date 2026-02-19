#ifndef INITIALIZELBM_H
#define INITIALIZELBM_H

#include "initializeLBM_inline.cuh"


void initialize_domain(nodeVar &dMom, nodeVar &hMom, haloData &gHalo, cylinderVar &h_cylinder, cylinderVar &d_cylinder);
__global__ void gpu_initialize_Moments_nodeType_GhostInterface(nodeVar dMom, haloData gHalo);

inline void allocateCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder, const size_t nb)
{
    // Host Memory
    checkCudaErrors(cudaMallocHost(&h_cylinder.boundaryList, nb * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.incomings, nb * Q * sizeof(binary_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.outgoings, nb * Q * sizeof(binary_t)));

    // Device Memory
    checkCudaErrors(cudaMalloc(&d_cylinder.boundaryList, nb * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.incomings, nb * Q * sizeof(binary_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.outgoings, nb * Q * sizeof(binary_t)));
}

inline void freeCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    // Free-ing host memory
    cudaFreeHost(h_cylinder.boundaryList);
    cudaFreeHost(h_cylinder.incomings);
    cudaFreeHost(h_cylinder.outgoings);

    // Free-ing device memory
    cudaFree(d_cylinder.boundaryList);
    cudaFree(d_cylinder.incomings);
    cudaFree(d_cylinder.outgoings);
}

inline void copyHostToDevice(cylinderVar &d_cylinder, cylinderVar &h_cylinder, const size_t nb)
{
    checkCudaErrors(cudaMemcpy(d_cylinder.boundaryList, h_cylinder.boundaryList, nb * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.incomings, h_cylinder.incomings, nb * Q * sizeof(binary_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.outgoings, h_cylinder.outgoings, nb * Q * sizeof(binary_t), cudaMemcpyHostToDevice));
}

#endif // INITIALIZELBM_H