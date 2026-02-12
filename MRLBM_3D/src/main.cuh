#ifndef MAIN_CUH
#define MAIN_CUH

#include "all_headers.h"
#include "solver/colrec/second_order/collision_streaming.cuh"
#include "save_data.cuh"
#include "postprocess.cuh"

// ---------------Host memory allocation----------------------
inline void allocateHostMemory(nodeVar &h_fMom)
{
    checkCudaErrors(cudaMallocHost(&h_fMom.nodeType, NUM_LBM_NODES * sizeof(nodeType_t)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.rho), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.ux), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.uy), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.mxx), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.myy), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.mxy), MEM_SIZE_LBM_NODES));
}

//---------------- Device Memory allocation------------------------
inline void allocateDeviceMemory(nodeVar &d_fMom)
{
    checkCudaErrors(cudaMalloc(&d_fMom.nodeType, NUM_LBM_NODES * sizeof(nodeType_t)));
    checkCudaErrors(cudaMalloc(&(d_fMom.rho), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.ux), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.uy), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.mxx), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.myy), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.mxy), MEM_SIZE_LBM_NODES));
}

inline void allocateHaloInterfaceMemory(haloData &fHalo_interface, haloData &gHalo_interface)
{
    checkCudaErrors(cudaMalloc(&fHalo_interface.X_WEST, NUM_HALO_FACE_X * QF * sizeof(real)));
    checkCudaErrors(cudaMalloc(&fHalo_interface.X_EAST, NUM_HALO_FACE_X * QF * sizeof(real)));
    checkCudaErrors(cudaMalloc(&fHalo_interface.Y_SOUTH, NUM_HALO_FACE_Y * QF * sizeof(real)));
    checkCudaErrors(cudaMalloc(&fHalo_interface.Y_NORTH, NUM_HALO_FACE_Y * QF * sizeof(real)));

    checkCudaErrors(cudaMalloc(&gHalo_interface.X_WEST, NUM_HALO_FACE_X * QF * sizeof(real)));
    checkCudaErrors(cudaMalloc(&gHalo_interface.X_EAST, NUM_HALO_FACE_X * QF * sizeof(real)));
    checkCudaErrors(cudaMalloc(&gHalo_interface.Y_SOUTH, NUM_HALO_FACE_Y * QF * sizeof(real)));
    checkCudaErrors(cudaMalloc(&gHalo_interface.Y_NORTH, NUM_HALO_FACE_Y * QF * sizeof(real)));
}

//---------------------------- Swap halo interface pointers: fHalo <--> gHalo
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
}

void copyHaloInterfaces(haloData &dst, const haloData &src)
{
    checkCudaErrors(cudaMemcpy(dst.X_WEST, src.X_WEST, sizeof(real) * NUM_HALO_FACE_X * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.X_EAST, src.X_EAST, sizeof(real) * NUM_HALO_FACE_X * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.Y_SOUTH, src.Y_SOUTH, sizeof(real) * NUM_HALO_FACE_Y * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.Y_NORTH, src.Y_NORTH, sizeof(real) * NUM_HALO_FACE_Y * QF, cudaMemcpyDeviceToDevice));
}

//-------------- Freeing host memory---------------------------
inline void freeHostMemory(nodeVar &h_fMom)
{
    cudaFreeHost(h_fMom.nodeType);
    cudaFreeHost(h_fMom.rho);
    cudaFreeHost(h_fMom.ux);
    cudaFreeHost(h_fMom.uy);
    cudaFreeHost(h_fMom.mxx);
    cudaFreeHost(h_fMom.myy);
    cudaFreeHost(h_fMom.mxy);
}

//--------------- Freeing device Memory------------------------
inline void freeDeviceMemory(nodeVar &d_fMom)
{
    cudaFree(d_fMom.nodeType);
    cudaFree(d_fMom.rho);
    cudaFree(d_fMom.ux);
    cudaFree(d_fMom.uy);
    cudaFree(d_fMom.mxx);
    cudaFree(d_fMom.myy);
    cudaFree(d_fMom.mxy);
}

inline void freeHaloInterfaceMemory(haloData &fHalo_interface, haloData &gHalo_interface)
{
    cudaFree(fHalo_interface.X_WEST);
    cudaFree(fHalo_interface.X_EAST);
    cudaFree(fHalo_interface.Y_SOUTH);
    cudaFree(fHalo_interface.Y_NORTH);
    cudaFree(gHalo_interface.X_WEST);
    cudaFree(gHalo_interface.X_EAST);
    cudaFree(gHalo_interface.Y_SOUTH);
    cudaFree(gHalo_interface.Y_NORTH);
}

// ------------------------Copy host --> Device--------------------------------
inline void copyMomentsHostToDevice(nodeVar &df_Mom, const nodeVar &h_fMom)
{
    checkCudaErrors(cudaMemcpy(df_Mom.rho, h_fMom.rho, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.ux, h_fMom.ux, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.uy, h_fMom.uy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.mxx, h_fMom.mxx, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.myy, h_fMom.myy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.mxy, h_fMom.mxy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
}

// ------------------------Copy Device --> Host--------------------------------
inline void copyMomentsDeviceToHost(nodeVar &h_fMom, const nodeVar &d_fMom)
{
    checkCudaErrors(cudaMemcpy(h_fMom.rho, d_fMom.rho, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.ux, d_fMom.ux, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.uy, d_fMom.uy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.mxx, d_fMom.mxx, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.myy, d_fMom.myy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.mxy, d_fMom.mxy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
}

inline void copyNodeTypeHostToDevice(nodeVar &d_fMom, const nodeVar &h_fMom)
{
    checkCudaErrors(cudaMemcpy(d_fMom.nodeType, h_fMom.nodeType, NUM_LBM_NODES * sizeof(nodeType_t),
                               cudaMemcpyHostToDevice));
}

void find_active_blocks(const nodeVar fMom, std::vector<int> &h_active_blocks)
{
    for (size_t by = 0; by < GRID_BLOCK_Y; ++by)
    {
        for (size_t bx = 0; bx < GRID_BLOCK_X; ++bx)
        {
            bool has_fluid = false;

            // Iterate through threads (nodes) in this block
            for (size_t ty = 0; ty < BLOCK_THREAD_Y; ++ty)
            {
                for (size_t tx = 0; tx < BLOCK_THREAD_X; ++tx)
                {
                    const unsigned int x = tx + bx * BLOCK_THREAD_X;
                    const unsigned int y = ty + by * BLOCK_THREAD_Y;

                    // Standard bounds check
                    if (x < NX && y < NY)
                    {
                        // Use your existing indexing macro
                        const size_t idx = IDX_BLOCK(tx, ty, bx, by);

                        // Check if node is NOT solid
                        if (fMom.nodeType[idx] != SOLID)
                        {
                            has_fluid = true;
                            break;
                        }
                    }
                }
                if (has_fluid)
                    break;
            }

            const size_t blockIdx = bx + by * GRID_BLOCK_X;
            h_active_blocks[blockIdx] = has_fluid ? 1 : 0;
        }
    }
}

#endif // MAIN_CUH