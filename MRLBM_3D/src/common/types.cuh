#pragma once

#include <cstdint>

typedef float real;
typedef uint16_t nodeType_t;
typedef uint8_t binary_t;

enum class MassBC : int
{
    Equilibrium = 0,
    Strong = 1,
    Weak = 2
};