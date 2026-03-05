#ifndef MAIN_CUH
#define MAIN_CUH


#include "saveData.cuh"

// ---------------Host memory allocation----------------------
inline void allocateHostMemory(nodeVar &h_fMom)
{
    checkCudaErrors(cudaMallocHost(&h_fMom.nodeType, NUM_LBM_NODES * sizeof(nodeType_t)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.rho), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.ux), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.uy), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.uz), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.mxx), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.myy), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.mzz), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.mxy), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.mxz), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.myz), NUM_LBM_NODES * sizeof(real)));
}

//---------------- Device Memory allocation------------------------
inline void allocateDeviceMemory(nodeVar &d_fMom)
{
    checkCudaErrors(cudaMalloc(&d_fMom.nodeType, NUM_LBM_NODES * sizeof(nodeType_t)));
    checkCudaErrors(cudaMalloc(&(d_fMom.rho), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.ux), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.uy), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.uz), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.mxx), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.myy), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.mzz), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.mxy), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.mxz), NUM_LBM_NODES * sizeof(real)));
    checkCudaErrors(cudaMalloc(&(d_fMom.myz), NUM_LBM_NODES * sizeof(real)));
}

__host__ __device__ inline void swapPointers(real *&pt1, real *&pt2)
{
    real *temp = pt1;
    pt1 = pt2;
    pt2 = temp;
}

// ------------------------Copy host --> Device--------------------------------
inline void copyMomentsHostToDevice(nodeVar &d_fMom, const nodeVar &h_fMom)
{
    checkCudaErrors(cudaMemcpy(d_fMom.rho, h_fMom.rho, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.ux, h_fMom.ux, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.uy, h_fMom.uy, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.uz, h_fMom.uz, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.mxx, h_fMom.mxx, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.myy, h_fMom.myy, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.mzz, h_fMom.mzz, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.mxy, h_fMom.mxy, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.mxz, h_fMom.mxz, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fMom.myz, h_fMom.myz, NUM_LBM_NODES * sizeof(real), cudaMemcpyHostToDevice));
}

// ------------------------Copy Device --> Host--------------------------------
inline void copyMomentsDeviceToHost(nodeVar &h_fMom, const nodeVar &d_fMom)
{
    checkCudaErrors(cudaMemcpy(h_fMom.rho, d_fMom.rho, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.ux, d_fMom.ux, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.uy, d_fMom.uy, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.uz, d_fMom.uz, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.mxx, d_fMom.mxx, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.myy, d_fMom.myy, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.mzz, d_fMom.mzz, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.mxy, d_fMom.mxy, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.mxz, d_fMom.mxz, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.myz, d_fMom.myz, NUM_LBM_NODES * sizeof(real), cudaMemcpyDeviceToHost));
}

inline void copyNodeTypeHostToDevice(nodeVar &d_fMom, const nodeVar &h_fMom)
{
    checkCudaErrors(cudaMemcpy(d_fMom.nodeType, h_fMom.nodeType, NUM_LBM_NODES * sizeof(nodeType_t),
                               cudaMemcpyHostToDevice));
}

#endif // MAIN_CUH