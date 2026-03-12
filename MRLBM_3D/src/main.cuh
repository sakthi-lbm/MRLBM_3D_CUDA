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

inline void allocateHaloInterfaceMemory(haloData &h_fHalo_interface, haloData &fHalo_interface,
                                        haloData &gHalo_interface)
{
    checkCudaErrors(cudaMalloc(&fHalo_interface.X_WEST, NUM_HALO_FACE_YZ * QF * sizeof(real)));  // x = 0
    checkCudaErrors(cudaMalloc(&fHalo_interface.X_EAST, NUM_HALO_FACE_YZ * QF * sizeof(real)));  // x = NX
    checkCudaErrors(cudaMalloc(&fHalo_interface.Y_SOUTH, NUM_HALO_FACE_XZ * QF * sizeof(real))); // y = 0
    checkCudaErrors(cudaMalloc(&fHalo_interface.Y_NORTH, NUM_HALO_FACE_XZ * QF * sizeof(real))); // y = NY
    checkCudaErrors(cudaMalloc(&fHalo_interface.Z_BACK, NUM_HALO_FACE_XY * QF * sizeof(real)));  // z = 0
    checkCudaErrors(cudaMalloc(&fHalo_interface.Z_FRONT, NUM_HALO_FACE_XY * QF * sizeof(real))); // z = NZ

    checkCudaErrors(cudaMalloc(&gHalo_interface.X_WEST, NUM_HALO_FACE_YZ * QF * sizeof(real)));  // x = 0
    checkCudaErrors(cudaMalloc(&gHalo_interface.X_EAST, NUM_HALO_FACE_YZ * QF * sizeof(real)));  // x = NX
    checkCudaErrors(cudaMalloc(&gHalo_interface.Y_SOUTH, NUM_HALO_FACE_XZ * QF * sizeof(real))); // y = 0
    checkCudaErrors(cudaMalloc(&gHalo_interface.Y_NORTH, NUM_HALO_FACE_XZ * QF * sizeof(real))); // y = NY
    checkCudaErrors(cudaMalloc(&gHalo_interface.Z_BACK, NUM_HALO_FACE_XY * QF * sizeof(real)));  // z = 0
    checkCudaErrors(cudaMalloc(&gHalo_interface.Z_FRONT, NUM_HALO_FACE_XY * QF * sizeof(real))); // z = NZ

    checkCudaErrors(cudaMallocHost(&h_fHalo_interface.X_WEST, NUM_HALO_FACE_YZ * QF * sizeof(real)));  // x = 0
    checkCudaErrors(cudaMallocHost(&h_fHalo_interface.X_EAST, NUM_HALO_FACE_YZ * QF * sizeof(real)));  // x = NX
    checkCudaErrors(cudaMallocHost(&h_fHalo_interface.Y_SOUTH, NUM_HALO_FACE_XZ * QF * sizeof(real))); // y = 0
    checkCudaErrors(cudaMallocHost(&h_fHalo_interface.Y_NORTH, NUM_HALO_FACE_XZ * QF * sizeof(real))); // y = NY
    checkCudaErrors(cudaMallocHost(&h_fHalo_interface.Z_BACK, NUM_HALO_FACE_XY * QF * sizeof(real)));  // z = 0
    checkCudaErrors(cudaMallocHost(&h_fHalo_interface.Z_FRONT, NUM_HALO_FACE_XY * QF * sizeof(real))); // z = NZ
}

__host__ __device__ inline void swapPointers(real *&pt1, real *&pt2)
{
    real *temp = pt1;
    pt1 = pt2;
    pt2 = temp;
}
inline void swapHaloInterfaces(haloData &fHalo, haloData &gHalo)
{
    swapPointers(fHalo.X_WEST, gHalo.X_WEST);
    swapPointers(fHalo.X_EAST, gHalo.X_EAST);
    swapPointers(fHalo.Y_SOUTH, gHalo.Y_SOUTH);
    swapPointers(fHalo.Y_NORTH, gHalo.Y_NORTH);
    swapPointers(fHalo.Z_BACK, gHalo.Z_BACK);
    swapPointers(fHalo.Z_FRONT, gHalo.Z_FRONT);
}

void copyHaloInterfaces(haloData &dst, const haloData &src)
{
    checkCudaErrors(cudaMemcpy(dst.X_WEST, src.X_WEST, sizeof(real) * NUM_HALO_FACE_YZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.X_EAST, src.X_EAST, sizeof(real) * NUM_HALO_FACE_YZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.Y_SOUTH, src.Y_SOUTH, sizeof(real) * NUM_HALO_FACE_XZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.Y_NORTH, src.Y_NORTH, sizeof(real) * NUM_HALO_FACE_XZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.Z_BACK, src.Z_BACK, sizeof(real) * NUM_HALO_FACE_XY * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.Z_FRONT, src.Z_FRONT, sizeof(real) * NUM_HALO_FACE_XY * QF, cudaMemcpyDeviceToDevice));
}

void copyHaloHostToDevice(haloData &dst, const haloData &src)
{
    checkCudaErrors(cudaMemcpy(dst.X_WEST, src.X_WEST, sizeof(real) * NUM_HALO_FACE_YZ * QF, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.X_EAST, src.X_EAST, sizeof(real) * NUM_HALO_FACE_YZ * QF, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.Y_SOUTH, src.Y_SOUTH, sizeof(real) * NUM_HALO_FACE_XZ * QF, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.Y_NORTH, src.Y_NORTH, sizeof(real) * NUM_HALO_FACE_XZ * QF, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.Z_BACK, src.Z_BACK, sizeof(real) * NUM_HALO_FACE_XY * QF, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.Z_FRONT, src.Z_FRONT, sizeof(real) * NUM_HALO_FACE_XY * QF, cudaMemcpyHostToDevice));
}

void copyHaloDeviceToHost(haloData &dst, const haloData &src)
{
    checkCudaErrors(cudaMemcpy(dst.X_WEST, src.X_WEST, sizeof(real) * NUM_HALO_FACE_YZ * QF, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dst.X_EAST, src.X_EAST, sizeof(real) * NUM_HALO_FACE_YZ * QF, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dst.Y_SOUTH, src.Y_SOUTH, sizeof(real) * NUM_HALO_FACE_XZ * QF, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dst.Y_NORTH, src.Y_NORTH, sizeof(real) * NUM_HALO_FACE_XZ * QF, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dst.Z_BACK, src.Z_BACK, sizeof(real) * NUM_HALO_FACE_XY * QF, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dst.Z_FRONT, src.Z_FRONT, sizeof(real) * NUM_HALO_FACE_XY * QF, cudaMemcpyDeviceToHost));
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

//-------------- Freeing host memory---------------------------
inline void freeHostMemory(nodeVar &h_fMom)
{
    cudaFreeHost(h_fMom.nodeType);
    cudaFreeHost(h_fMom.rho);
    cudaFreeHost(h_fMom.ux);
    cudaFreeHost(h_fMom.uy);
    cudaFreeHost(h_fMom.uz);
    cudaFreeHost(h_fMom.mxx);
    cudaFreeHost(h_fMom.myy);
    cudaFreeHost(h_fMom.mzz);
    cudaFreeHost(h_fMom.mxy);
    cudaFreeHost(h_fMom.mxz);
    cudaFreeHost(h_fMom.myz);
}

//--------------- Freeing device Memory------------------------
inline void freeDeviceMemory(nodeVar &d_fMom)
{
    cudaFree(d_fMom.nodeType);
    cudaFree(d_fMom.rho);
    cudaFree(d_fMom.ux);
    cudaFree(d_fMom.uy);
    cudaFree(d_fMom.uz);
    cudaFree(d_fMom.mxx);
    cudaFree(d_fMom.myy);
    cudaFree(d_fMom.mzz);
    cudaFree(d_fMom.mxy);
    cudaFree(d_fMom.mxz);
    cudaFree(d_fMom.myz);
}

inline void freeHaloInterfaceMemory(haloData &h_fHalo_interface, haloData &fHalo_interface, haloData &gHalo_interface)
{
    cudaFree(fHalo_interface.X_WEST);
    cudaFree(fHalo_interface.X_EAST);
    cudaFree(fHalo_interface.Y_SOUTH);
    cudaFree(fHalo_interface.Y_NORTH);
    cudaFree(fHalo_interface.Z_BACK);
    cudaFree(fHalo_interface.Z_FRONT);

    cudaFree(gHalo_interface.X_WEST);
    cudaFree(gHalo_interface.X_EAST);
    cudaFree(gHalo_interface.Y_SOUTH);
    cudaFree(gHalo_interface.Y_NORTH);
    cudaFree(gHalo_interface.Z_BACK);
    cudaFree(gHalo_interface.Z_FRONT);

    cudaFree(h_fHalo_interface.X_WEST);
    cudaFree(h_fHalo_interface.X_EAST);
    cudaFree(h_fHalo_interface.Y_SOUTH);
    cudaFree(h_fHalo_interface.Y_NORTH);
    cudaFree(h_fHalo_interface.Z_BACK);
    cudaFree(h_fHalo_interface.Z_FRONT);
}

#endif // MAIN_CUH