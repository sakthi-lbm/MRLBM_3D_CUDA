#pragma once

// ================= CASE SELECTION =================
#define CASE_CYLINDER 1
#define CASE_AIRFOIL 2
#define CASE_ANNULUS 3

#define CASE_ID CASE_CYLINDER

#define REG_ORDER second_order
#define STENCIL D3Q27

#define GPU_INDEX 0

// host-only timing type
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
typedef std::chrono::high_resolution_clock::time_point timestep;

// clang-format off

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

// ================= PATH SYSTEM =================
#define CASE_DIRECTORY cases
#define CORE_DIRECTORY core
#define SOLVER_DIRECTORY solver
#define UTILS_DIRECTORY utils


#if CASE_ID == CASE_CYLINDER
#define CASE_NAME cylinder
#elif CASE_ID == CASE_AIRFOIL
#define CASE_NAME airfoil
#elif CASE_ID == CASE_ANNULUS
#define CASE_NAME annulus
#else
#error "Invalid CASE_ID"
#endif

#define CASE_PATH CASE_DIRECTORY/CASE_NAME
#define STENCIL_PATH CORE_DIRECTORY/STENCIL

#define LATTICE_PROPERTIES STR(STENCIL_PATH/latticeProperties.cuh)

#define CASE_BOUNDARY STR(CASE_PATH/boundaries.cuh)
#define CASE_OUTPUTS STR(CASE_PATH/outputs.cuh)
#define RECONSTRUCT STR(STENCIL_PATH/REG_ORDER/reconstruction.cuh)
#define STREAMING STR(CORE_DIRECTORY/streaming.cuh)

#define FILE_UTILS STR(UTILS_DIRECTORY/file_utils.cuh)


//Case  selection
#if CASE_ID == CASE_CYLINDER
    #define CASE_KERNELS STR(CASE_DIRECTORY/cylinder/cylinder_kernels.cuh)

#elif CASE_ID == CASE_AIRFOIL
    #define CASE_KERNELS STR(CASE_DIRECTORY/airfoil/airfoil_kernels.cuh)

#elif CASE_ID == CASE_ANNULUS
    #define CASE_KERNELS STR(CASE_DIRECTORY/annulus/annulus_kernels.cuh)

#else
    #error "Unknown CASE_ID"
#endif

// clang-format on
