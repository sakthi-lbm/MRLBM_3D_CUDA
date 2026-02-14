#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <chrono>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// ================= Problem Definition =================
#define BC_PROBLEM cylinder
#define REG_ORDER second_order
#define STENCIL D3Q27


// ================= Types =================

typedef float real;
typedef uint16_t nodeType_t;
typedef uint8_t binary_t;

// host-only timing type
#include <chrono>
typedef std::chrono::high_resolution_clock::time_point timestep;

// ================= Constants =================
constexpr real PI     = real(3.14159265358979323846);
constexpr real SQRT_2 = real(1.4142135623730950488);
constexpr real SQRT_3 = real(1.7320508075688772935);


template <typename T>
__host__ __device__ inline constexpr real toReal(T value)
{
    return static_cast<real>(value);
}

template <typename T>
__host__ __device__ inline constexpr int toInt(T value)
{
    return static_cast<int>(value);
}

template <typename T>
__host__ __device__ inline constexpr size_t toSize_t(T value)
{
    return static_cast<size_t>(value);
}

template <typename T>
__host__ __device__ inline constexpr float toFloat(T value)
{
    return static_cast<float>(value);
}

template <typename T>
__host__ __device__ inline constexpr nodeType_t toNodeTypeT(T value)
{
    return static_cast<nodeType_t>(value);
}

// ================= Enums =================

enum class MassBC : int
{
    Equilibrium = 0,
    Strong = 1,
    Weak = 2
};

template <typename... Args>
std::string construct_path(Args &&...parts)
{
    std::filesystem::path p;
    (p /= ... /= parts);
    return p.string();
}

constexpr size_t MAX_THREADS_PER_BLOCK = 1024;
constexpr dim3 findOptimalBlockDim3D(size_t maxSharedMemBytes, size_t bytesPerThread)
{
    size_t bestX = 1;
    size_t bestY = 1;
    size_t bestZ = 1;
    size_t maxThreadsUsed = 0;

    for (size_t dimZ = 1; dimZ <= 32; dimZ *= 2)
    {
        for (size_t dimY = 1; dimY <= 32; dimY *= 2)
        {
            for (size_t dimX = 1; dimX <= 32; dimX *= 2)
            {
                size_t threads = dimX * dimY * dimZ;
                size_t usedMem = threads * bytesPerThread;

                if (threads > MAX_THREADS_PER_BLOCK)
                    continue;

                if (usedMem <= maxSharedMemBytes)
                {
                    if (threads > maxThreadsUsed)
                    {
                        bestX = dimX;
                        bestY = dimY;
                        bestZ = dimZ;
                        maxThreadsUsed = threads;
                    }
                }
            }
        }
    }

    return dim3(bestX, bestY, bestZ);
}

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

#define CASE_DIRECTORY cases
#define COLREC_DIRECTORY colrec
#define SOLVER_DIRECTORY solver

#define CASE_PATH CASE_DIRECTORY/BC_PROBLEM
#define STENCIL_PATH SOLVER_DIRECTORY/COLREC_DIRECTORY/STENCIL

#define LATTICE_PROPERTIES STR(STENCIL_PATH/latticeProperties.cuh)
#define HALO_INTERFACE STR(STENCIL_PATH/halo_interface.cuh)

#define CASE_CONSTANTS STR(CASE_PATH/constants.h)
#define CASE_BOUNDARY STR(CASE_PATH/boundaries.cuh)
#define CASE_OUTPUTS STR(CASE_PATH/outputs.h)
#define RECONSTRUCT STR(STENCIL_PATH/REG_ORDER/reconstruction.cuh)
#define STREAMING STR(STENCIL_PATH/streaming.cuh)
#define EVAL_MOMENTS STR(STENCIL_PATH/eval_moments.cuh)




#endif
