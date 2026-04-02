#pragma once

#include <cstdint>

using  real = float;
using  nodeType_t = uint32_t;
using  binary_t = uint8_t;

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

enum MaskType
{
    INCOMING = 0,
    OUTGOING = 1
};

