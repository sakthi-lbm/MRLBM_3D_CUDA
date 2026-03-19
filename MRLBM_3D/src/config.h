#pragma once

// ================= Problem Definition =================
#define BC_PROBLEM cylinder
#define REG_ORDER second_order
#define STENCIL D3Q27

// host-only timing type
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
typedef std::chrono::high_resolution_clock::time_point timestep;

// clang-format off

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

#define CASE_DIRECTORY cases
#define CORE_DIRECTORY core
#define SOLVER_DIRECTORY solver

#define CASE_PATH CASE_DIRECTORY/BC_PROBLEM
#define STENCIL_PATH CORE_DIRECTORY/STENCIL

#define LATTICE_PROPERTIES STR(STENCIL_PATH/latticeProperties.cuh)

#define CASE_BOUNDARY STR(CASE_PATH/boundaries.cuh)
#define CASE_OUTPUTS STR(CASE_PATH/outputs.cuh)
#define RECONSTRUCT STR(STENCIL_PATH/REG_ORDER/reconstruction.cuh)
#define STREAMING STR(CORE_DIRECTORY/streaming.cuh)

// clang-format on


