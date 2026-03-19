#pragma once
#include <filesystem>

#include "types.cuh"

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

__host__ __device__ inline constexpr real rabs(real x)
{
    return (x < real(0)) ? -x : x;
}

template <typename... Args>
std::string construct_path(Args &&...parts)
{
    std::filesystem::path p;
    (p /= ... /= parts);
    return p.string();
}
