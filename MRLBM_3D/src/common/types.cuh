#pragma once

#include <cstdint>

typedef float real;
typedef uint64_t nodeType_t;
typedef uint8_t binary_t;

// ================= GPU BLOCK CONFIG =================
constexpr size_t BLOCK_THREAD_X = 8;
constexpr size_t BLOCK_THREAD_Y = 8;
constexpr size_t BLOCK_THREAD_Z = 4;

enum class MassBC : int
{
    Equilibrium = 0,
    Strong = 1,
    Weak = 2
};