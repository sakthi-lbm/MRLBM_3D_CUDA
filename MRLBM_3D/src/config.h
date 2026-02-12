#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream> // std::cout, std::fixed
#include <iomanip>  // std::setprecision
#include <filesystem>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

// ======================= PROBLEM DEFINITION =====================================================

#define BC_PROBLEM cylinder
#define REG_ORDER second_order

// ==================================================================================================

typedef double real;
typedef uint16_t nodeType_t;
typedef uint8_t binary_t;
typedef std::chrono::high_resolution_clock::time_point timestep;

constexpr real PI = 3.14159265358979323846;
constexpr real SQRT_2 = 1.4142135623730950488;
constexpr real SQRT_3 = 1.7320508075688772935;

#define GPU_INDEX 0

template <typename T>
__host__ __device__ inline constexpr real toReal(const T value)
{
    return static_cast<real>(value);
}
template <typename T>
__host__ __device__ inline constexpr int toInt(const T value)
{
    return static_cast<int>(value);
}

template <typename T>
__host__ __device__ inline constexpr size_t toSize_t(const T value)
{
    return static_cast<size_t>(value);
}

template <typename T>
__host__ __device__ inline constexpr float toFloat(const T value)
{
    return static_cast<float>(value);
}

template <typename T>
__host__ __device__ inline constexpr nodeType_t toNodeTypeT(const T value)
{
    return static_cast<nodeType_t>(value);
}

__host__ __device__ inline constexpr real rabs(real x)
{
    return (x < real(0)) ? -x : x;
}

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

// clang-format off

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

#define CASE_DIRECTORY cases
#define COLREC_DIRECTORY colrec
#define SOLVER_DIRECTORY solver

#define CASE_CONSTANTS STR(CASE_DIRECTORY/BC_PROBLEM/constants.h)
#define CASE_OUTPUTS STR(CASE_DIRECTORY/BC_PROBLEM/outputs.h)
#define RECONSTRUCT STR(SOLVER_DIRECTORY/COLREC_DIRECTORY/REG_ORDER/reconstruction.cuh)
#define CASE_BOUNDARY STR(CASE_DIRECTORY/BC_PROBLEM/boundaries.cuh)
#define CASE_POST STR(CASE_DIRECTORY/BC_PROBLEM/post_case.cuh)

// clang-format on

#endif // CONFIG_H