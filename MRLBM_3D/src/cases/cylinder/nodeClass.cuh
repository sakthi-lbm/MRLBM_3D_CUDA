#pragma once
#include<iostream>

#include "../../config.h"
#include STREAMING

inline bool check_for_neighbour_type(const nodeType_t node[Q], const nodeType_t TYPE)
{
    for (int q = 1; q < Q; q++)
        if (node[q] == TYPE)
            return true;

    return false;
}

inline size_t idx3D(int x, int y, int z)
{
    return IDX_BLOCK(x % BLOCK_THREAD_X,
                     y % BLOCK_THREAD_Y,
                     z % BLOCK_THREAD_Z,
                     x / BLOCK_THREAD_X,
                     y / BLOCK_THREAD_Y,
                     z / BLOCK_THREAD_Z);
}

inline void load_neighbors(nodeType_t node[Q], const nodeVar &hMom, int x, int y, int z)
{
    for (int q = 0; q < Q; q++)
    {
        int xn = x + h_cx[q];
        int yn = y + h_cy[q];
        int zn = z + h_cz[q];

#if Z_PERIODIC
        if (zn < 0)
            zn += NZ;
        else if (zn >= NZ)
            zn -= NZ;

        node[q] = hMom.nodeType[idx3D(xn, yn, zn)];
#else
        if (zn < 0 || zn >= NZ)
            node[q] = SOLID;
        else
            node[q] = hMom.nodeType[idx3D(xn, yn, zn)];
#endif
    }
}

inline int compute_bits_type(const nodeType_t node[Q], binary_t bits[8], const nodeType_t TYPE)
{
    constexpr int NUM_BITS = 8;
    constexpr int DIRS_PER_BIT = 7;

    constexpr int bit_dirs[NUM_BITS][DIRS_PER_BIT] = {
        {2, 4, 6, 8, 10, 12, 20},
        {1, 4, 6, 12, 13, 15, 26},
        {2, 3, 6, 10, 14, 17, 24},
        {1, 3, 6, 7, 15, 17, 21},
        {2, 4, 5, 8, 16, 18, 22},
        {1, 4, 5, 9, 13, 18, 23},
        {2, 3, 5, 11, 14, 16, 25},
        {1, 3, 5, 7, 9, 11, 19}};

    int bit_count = 0;

    for (int b = 0; b < NUM_BITS; b++)
    {
        bits[b] = 0;

        for (int i = 0; i < DIRS_PER_BIT; i++)
        {
            if (node[bit_dirs[b][i]] == TYPE)
            {
                bits[b] = 1;
                break;
            }
        }

        bit_count += bits[b];
    }

    return bit_count;
}

inline int compute_node_tag(const binary_t bits[8])
{
    return bits[0] * 1 +
           bits[1] * 2 +
           bits[2] * 4 +
           bits[3] * 8 +
           bits[4] * 16 +
           bits[5] * 32 +
           bits[6] * 64 +
           bits[7] * 128;
}

inline void mark_cylinder_solid(nodeVar &hMom)
{
    for (int z = 0; z < NZ; z++)
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
            {
                const real x_diff = toReal(x) - XC;
                const real y_diff = toReal(y) - YC;
                const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

                const size_t idx = idx3D(x, y, z);

                if (radius <= R_WALL)
                    hMom.nodeType[idx] = SOLID;
            }
}

inline void assign_tagged_node(nodeVar &hMom,
                               int x, int y, int z,
                               nodeType_t baseType,
                               const binary_t bits[8])
{
    const int node_tag = compute_node_tag(bits);
    const size_t idx = idx3D(x, y, z);

    hMom.nodeType[idx] = baseType + toNodeTypeT(node_tag);
}

inline int classify_boundary_nodes_staircase(nodeVar &hMom)
{
    constexpr int X_BEGIN = LW - 2;
    constexpr int X_END = LW + D + 2;

    constexpr int Y_BEGIN = LS - 2;
    constexpr int Y_END = LS + D + 2;

    int count = 0;
    nodeType_t node[Q];
    for (int z = 0; z < NZ; z++)
        for (int y = Y_BEGIN; y < Y_END; y++)
            for (int x = X_BEGIN; x < X_END; x++)
            {
                load_neighbors(node, hMom, x, y, z);
                bool anyFluid = check_for_neighbour_type(node, BULK); // check for any FLUID neighbour

                if (node[0] == SOLID && anyFluid)
                {
                    binary_t bits[8] = {0};
                    const int bit_count = compute_bits_type(node, bits, BULK);
                    assign_tagged_node(hMom, x, y, z, INNER_NODE, bits);
                    count++;
                }
            }
    return count;
}

inline int classify_boundary_nodes_triangular(nodeVar &hMom)
{
    constexpr int X_BEGIN = LW - 2;
    constexpr int X_END = LW + D + 2;

    constexpr int Y_BEGIN = LS - 2;
    constexpr int Y_END = LS + D + 2;

    int count = 0;
    nodeType_t node[Q];
    for (int z = 0; z < NZ; z++)
        for (int y = Y_BEGIN; y < Y_END; y++)
            for (int x = X_BEGIN; x < X_END; x++)
            {
                load_neighbors(node, hMom, x, y, z);
                bool anyFluid = check_for_neighbour_type(node, BULK); // check for any FLUID neighbour

                if (node[0] == SOLID && anyFluid)
                {
                    binary_t bits[8] = {0};
                    const int bit_count = compute_bits_type(node, bits, BULK);
#if Z_PERIODIC
                    if (bit_count > 2)
                    {
                        assign_tagged_node(hMom, x, y, z, INNER_NODE, bits);
                        count++;
                    }
#else
                    if (((z == 0 || z == NZ - 1) && bit_count > 1) ||
                        ((z > 0 && z < NZ - 1) && bit_count > 2))
                    {
                        assign_tagged_node(hMom, x, y, z, INNER_NODE, bits);
                        count++;
                    }
#endif
                }
            }
    return count;
}

inline int classify_bcfluid_nodes_triangular(nodeVar &hMom)
{
    constexpr int Z_BEGIN = (Z_PERIODIC ? 0 : 1);
    constexpr int Z_END = (Z_PERIODIC ? NZ : NZ - 1);

    constexpr int X_BEGIN = LW - 2;
    constexpr int X_END = LW + D + 2;

    constexpr int Y_BEGIN = LS - 2;
    constexpr int Y_END = LS + D + 2;

    int count = 0;
    nodeType_t node[Q];
    for (int z = Z_BEGIN; z < Z_END; z++)
        for (int y = Y_BEGIN; y < Y_END; y++)
            for (int x = X_BEGIN; x < X_END; x++)
            {
                load_neighbors(node, hMom, x, y, z);
                bool anySolid = check_for_neighbour_type(node, SOLID); // finding for any SOLID neighbour

                if (node[0] == BULK && anySolid)
                {
                    binary_t bits[8] = {0};
                    const int bit_count = compute_bits_type(node, bits, SOLID);
                    assign_tagged_node(hMom, x, y, z, BCFLUID_NODE, bits);
                    count++;
                }
            }
    return count;
}

inline int classify_bcsolid_nodes_triangular(nodeVar &hMom)
{
    constexpr int X_BEGIN = LW - 2;
    constexpr int X_END = LW + D + 2;

    constexpr int Y_BEGIN = LS - 2;
    constexpr int Y_END = LS + D + 2;

    int count = 0;
    nodeType_t node[Q];
    for (int z : {0, NZ - 1})
        for (int y = Y_BEGIN; y < Y_END; y++)
            for (int x = X_BEGIN; x < X_END; x++)
            {
                load_neighbors(node, hMom, x, y, z);

                if (node[0] == BACK || node[0] == FRONT)
                {
                    binary_t bits[8] = {0};
                    const int bit_count = compute_bits_type(node, bits, BULK);

                    if (bit_count < 4)
                    {
                        assign_tagged_node(hMom, x, y, z, BCSOLID_NODE, bits);
                        count++;
                    }
                }
            }
    return count;
}

inline void initialize_cylinder_nodeType_staircase(nodeVar &hMom)
{
    static_assert(LW >= 3 && LS >= 3);
    static_assert(LW + D + 3 < NX);
    static_assert(LS + D + 3 < NY);

    mark_cylinder_solid(hMom);

    NB = classify_boundary_nodes_staircase(hMom);
    std::cout << "inner: " << NB << std::endl;
}

inline void initialize_cylinder_nodeType_triangular(nodeVar &hMom)
{
    static_assert(LW >= 3 && LS >= 3);
    static_assert(LW + D + 3 < NX);
    static_assert(LS + D + 3 < NY);

    mark_cylinder_solid(hMom);

    NB = classify_boundary_nodes_triangular(hMom);
    std::cout << "Boundary nodes: " << NB << std::endl;

    NB_FLUID = classify_bcfluid_nodes_triangular(hMom);
    std::cout << "BcFluid nodes: " << NB_FLUID << std::endl;

#if !Z_PERIODIC
    NB_SOLID = classify_bcsolid_nodes_triangular(hMom);
    std::cout << "BcSolid nodes: " << NB_SOLID << std::endl;
#endif
}