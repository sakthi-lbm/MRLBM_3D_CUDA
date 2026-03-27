#pragma once

#include "utils.cuh"

/*   N
+----+----+
| 64 |128 |
W  +----+----+  E      (BACK, z=0)
| 16 | 32 |
+----+----+
     S
/*
/*       N
+----+----+
|  4 |  8 |
W  +----+----+  E      (FRONT, z=NZ)
|  1 |  2 |
+----+----+
     S
*/

// SOLID combinations:
constexpr nodeType_t TAG_BCF_1 = 17;
constexpr nodeType_t TAG_BCF_2 = 34;
constexpr nodeType_t TAG_BCF_3 = 68;
constexpr nodeType_t TAG_BCF_4 = 136;

// FLUID combinations:
constexpr nodeType_t TAG_BCS_1 = 7;
constexpr nodeType_t TAG_BCS_2 = 11;
constexpr nodeType_t TAG_BCS_3 = 13;
constexpr nodeType_t TAG_BCS_4 = 14;
constexpr nodeType_t TAG_BCS_5 = 112;
constexpr nodeType_t TAG_BCS_6 = 176;
constexpr nodeType_t TAG_BCS_7 = 208;
constexpr nodeType_t TAG_BCS_8 = 224;

enum NodeTypeEnum : nodeType_t
{
    NODE_BULK = 0,
    NODE_SOLID,

    NODE_INNER,
    NODE_OUTER,

    NODE_BCFLUID_INNER,
    NODE_BCFLUID_OUTER,

    NODE_BCSOLID_INNER,
    NODE_BCSOLID_OUTER,

    NODE_NORTH,
    NODE_SOUTH,
    NODE_EAST,
    NODE_WEST,
    NODE_FRONT,
    NODE_BACK,

    NODE_NORTH_EAST,
    NODE_NORTH_WEST,
    NODE_NORTH_FRONT,
    NODE_NORTH_BACK,
    NODE_SOUTH_EAST,
    NODE_SOUTH_WEST,
    NODE_SOUTH_FRONT,
    NODE_SOUTH_BACK,
    NODE_EAST_FRONT,
    NODE_EAST_BACK,
    NODE_WEST_FRONT,
    NODE_WEST_BACK,

    NODE_NORTH_WEST_FRONT,
    NODE_NORTH_WEST_BACK,
    NODE_SOUTH_WEST_FRONT,
    NODE_SOUTH_WEST_BACK,
    NODE_NORTH_EAST_FRONT,
    NODE_NORTH_EAST_BACK,
    NODE_SOUTH_EAST_FRONT,
    NODE_SOUTH_EAST_BACK,
};

constexpr int TYPE_SHIFT = 20;
constexpr nodeType_t INDEX_MASK = (1u << TYPE_SHIFT) - 1;

__host__ __device__ inline nodeType_t encodeNode(nodeType_t type, nodeType_t idx)
{
    return (type << TYPE_SHIFT) | idx;
}

__host__ __device__ inline nodeType_t getType(nodeType_t nodeType)
{
    return nodeType >> TYPE_SHIFT;
}

__host__ __device__ inline nodeType_t getIndex(nodeType_t nodeType)
{
    return nodeType & INDEX_MASK;
}