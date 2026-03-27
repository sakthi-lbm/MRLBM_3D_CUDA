#pragma once
#include <iostream>

#include "annulus_structs.h"

inline bool check_for_neighbour_type(const nodeType_t *node, const nodeType_t TYPE)
{
    for (int q = 1; q < Q; q++)
        if (getType(node[q]) == TYPE)
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

inline void load_neighbors(nodeType_t *node, const nodeVar &hMom, int x, int y, int z)
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

        if (xn < 0 || xn >= NX || yn < 0 || yn >= NY)
        {
            node[q] = encodeNode(NODE_SOLID, 0);
        }
        else
        {
            node[q] = hMom.nodeType[idx3D(xn, yn, zn)];
        }
#else
        if (xn < 0 || xn >= NX || yn < 0 || yn >= NY || zn < 0 || zn >= NZ)
        {
            node[q] = encodeNode(NODE_SOLID, 0);
        }
        else
        {
            node[q] = hMom.nodeType[idx3D(xn, yn, zn)];
        }
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
            if (getType(node[bit_dirs[b][i]]) == TYPE)
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

inline void mark_annulus_solid(nodeVar &hMom)
{
    for (int z = 0; z < NZ; z++)
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
            {
                const real x_diff = toReal(x) - XC;
                const real y_diff = toReal(y) - YC;
                const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);
                const real r_inner = R_IN;
                const real r_outer = R_OUT;

                const size_t idx = idx3D(x, y, z);

                if (radius <= r_inner || radius >= r_outer)
                    hMom.nodeType[idx] = encodeNode(NODE_SOLID, 0);
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

inline void classify_boundary_nodes_staircase(nodeVar &hMom, int &nb_inner, int &nb_outer)
{
    constexpr int Z_BEGIN = (Z_PERIODIC ? 0 : 1);
    constexpr int Z_END = (Z_PERIODIC ? NZ : NZ - 1);

    constexpr int X_BEGIN = 0;
    constexpr int X_END = NX;

    constexpr int Y_BEGIN = 0;
    constexpr int Y_END = NY;

    real R_mid = 0.5 * toReal(R_IN + R_OUT);
    int inner_count = 0;
    int outer_count = 0;
    nodeType_t node[Q];

    for (int z = Z_BEGIN; z < Z_END; z++)
        for (int y = Y_BEGIN; y < Y_END; y++)
            for (int x = X_BEGIN; x < X_END; x++)
            {
                load_neighbors(node, hMom, x, y, z);
                bool anyFluid = check_for_neighbour_type(node, NODE_BULK); // check for any FLUID neighbour

                if (getType(node[0]) == NODE_SOLID && anyFluid)
                {
                    const real x_diff = toReal(x) - XC;
                    const real y_diff = toReal(y) - YC;
                    const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

                    const size_t idx = idx3D(x, y, z);

                    if (radius < R_mid)
                    {
                        hMom.nodeType[idx] = encodeNode(NODE_INNER, 0);
                        inner_count++;
                    }
                    else
                    {
                        hMom.nodeType[idx] = encodeNode(NODE_OUTER, 0);
                        outer_count++;
                    }
                }
            }
    nb_inner = inner_count;
    nb_outer = outer_count;
}

inline void classify_boundary_nodes_triangular(nodeVar &hMom, int &nb_inner, int &nb_outer)
{
    constexpr int Z_BEGIN = (Z_PERIODIC ? 0 : 1);
    constexpr int Z_END = (Z_PERIODIC ? NZ : NZ - 1);

    constexpr int X_BEGIN = 0;
    constexpr int X_END = NX;

    constexpr int Y_BEGIN = 0;
    constexpr int Y_END = NY;

    const real R_mid = 0.5 * toReal(R_IN + R_OUT);
    int inner_count = 0;
    int outer_count = 0;
    nodeType_t node[Q];

    for (int z = Z_BEGIN; z < Z_END; z++)
        for (int y = Y_BEGIN; y < Y_END; y++)
            for (int x = X_BEGIN; x < X_END; x++)
            {
                if (x >= NX_phys || y >= NY_phys || z >= NZ_phys)
                    continue;

                load_neighbors(node, hMom, x, y, z);
                bool anyFluid = check_for_neighbour_type(node, NODE_BULK); // check for any FLUID neighbour

                if (getType(node[0]) == NODE_SOLID && anyFluid)
                {
                    binary_t bits[8] = {0};
                    const int bit_count = compute_bits_type(node, bits, NODE_BULK);
                    const real x_diff = toReal(x) - XC;
                    const real y_diff = toReal(y) - YC;
                    const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);
                    const size_t idx = idx3D(x, y, z);

#if Z_PERIODIC
                    if (bit_count > 2)
                    {
                        if (radius < R_mid)
                        {
                            hMom.nodeType[idx] = encodeNode(NODE_INNER, 0);
                            inner_count++;
                        }
                        else
                        {
                            hMom.nodeType[idx] = encodeNode(NODE_OUTER, 0);
                            outer_count++;
                        }
                    }
#else
                    if (((z == 0 || z == NZ - 1) && bit_count > 1) ||
                        ((z > 0 && z < NZ - 1) && bit_count > 2))
                    {
                        if (radius < R_mid)
                        {
                            hMom.nodeType[idx] = encodeNode(NODE_INNER, 0);
                            inner_count++;
                        }
                        else
                        {
                            hMom.nodeType[idx] = encodeNode(NODE_OUTER, 0);
                            outer_count++;
                        }
                    }
#endif
                }
            }
    nb_inner = inner_count;
    nb_outer = outer_count;
}

inline void classify_bcfluid_nodes_triangular(nodeVar &hMom, int &nb_fluid_inner, int &nb_fluid_outer)
{
    constexpr int Z_BEGIN = (Z_PERIODIC ? 0 : 1);
    constexpr int Z_END = (Z_PERIODIC ? NZ : NZ - 1);

    constexpr int X_BEGIN = 0;
    constexpr int X_END = NX;

    constexpr int Y_BEGIN = 0;
    constexpr int Y_END = NY;

    const real R_mid = toReal(0.5) * (R_IN + R_OUT);
    int inner_count = 0;
    int outer_count = 0;
    nodeType_t node[Q];
    for (int z = Z_BEGIN; z < Z_END; z++)
        for (int y = Y_BEGIN; y < Y_END; y++)
            for (int x = X_BEGIN; x < X_END; x++)
            {
                load_neighbors(node, hMom, x, y, z);
                bool anySolid = check_for_neighbour_type(node, NODE_SOLID); // finding for any SOLID neighbour

                if (getType(node[0]) == NODE_BULK && anySolid)
                {
                    const real x_diff = toReal(x) - XC;
                    const real y_diff = toReal(y) - YC;
                    const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);
                    const size_t idx = idx3D(x, y, z);

                    binary_t bits[8] = {0};
                    const int bit_count = compute_bits_type(node, bits, NODE_SOLID);
                    const nodeType_t tag = toNodeTypeT(bit_count);

                    if (radius < R_mid)
                    {
                        hMom.nodeType[idx] = encodeNode(NODE_BCFLUID_INNER, tag);
                        inner_count++;
                    }
                    else
                    {
                        hMom.nodeType[idx] = encodeNode(NODE_BCFLUID_OUTER, tag);
                        outer_count++;
                    }
                }
            }
    nb_fluid_inner = inner_count;
    nb_fluid_outer = outer_count;
}

inline void classify_bcsolid_nodes_triangular(nodeVar &hMom, int &nb_solid_inner, int &nb_solid_outer)
{
    constexpr int X_BEGIN = 0;
    constexpr int X_END = NX;

    constexpr int Y_BEGIN = 0;
    constexpr int Y_END = NY;

    real R_mid = toReal(0.5) * (R_IN + R_OUT);
    int inner_count = 0;
    int outer_count = 0;
    nodeType_t node[Q];
    for (int z : {0, NZ - 1})
        for (int y = Y_BEGIN; y < Y_END; y++)
            for (int x = X_BEGIN; x < X_END; x++)
            {
                load_neighbors(node, hMom, x, y, z);

                if (getType(node[0]) == NODE_BACK || getType(node[0]) == NODE_FRONT)
                {
                    const real x_diff = toReal(x) - XC;
                    const real y_diff = toReal(y) - YC;
                    const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

                    const size_t idx = idx3D(x, y, z);

                    binary_t bits[8] = {0};
                    const int bit_count = compute_bits_type(node, bits, NODE_BULK);
                    const nodeType_t tag = toNodeTypeT(bit_count);

                    if (bit_count < 4)
                    {
                        if (radius < R_mid)
                        {
                            hMom.nodeType[idx] = encodeNode(NODE_BCSOLID_INNER, tag);
                            inner_count++;
                        }
                        else
                        {
                            hMom.nodeType[idx] = encodeNode(NODE_BCSOLID_OUTER, tag);
                            outer_count++;
                        }
                    }
                }
            }
    nb_solid_inner = inner_count;
    nb_solid_outer = outer_count;
}

inline void initialize_annulus_nodeType_staircase(nodeVar &hMom, boundaryVar &annulus_inner,
                                                  boundaryVar &annulus_outer)
{
    mark_annulus_solid(hMom);

    int nb_inner, nb_outer;
    classify_boundary_nodes_staircase(hMom, nb_inner, nb_outer);
    annulus_inner.NB = nb_inner;
    annulus_outer.NB = nb_outer;
    std::cout << "inner: " << nb_inner << std::endl;
    std::cout << "outer: " << nb_outer << std::endl;
}

inline void initialize_annulus_nodeType_triangular(nodeVar &hMom, boundaryVar &annulus_inner,
                                                   boundaryVar &annulus_outer)
{
    mark_annulus_solid(hMom);

    int nb_inner, nb_outer;
    classify_boundary_nodes_triangular(hMom, nb_inner, nb_outer);

    annulus_inner.NB = nb_inner;
    annulus_outer.NB = nb_outer;
    std::cout << "inner: " << nb_inner << std::endl;
    std::cout << "outer: " << nb_outer << std::endl;

    int nb_fluid_inner, nb_fluid_outer;
    classify_bcfluid_nodes_triangular(hMom, nb_fluid_inner, nb_fluid_outer);
    annulus_inner.NB_FLUID = nb_fluid_inner;
    annulus_outer.NB_FLUID = nb_fluid_outer;
    std::cout << "BcFluid inner nodes: " << nb_fluid_inner << std::endl;
    std::cout << "BcFluid outer nodes: " << nb_fluid_outer << std::endl;

#if !Z_PERIODIC

    int nb_solid_inner, nb_solid_outer;
    classify_bcsolid_nodes_triangular(hMom, nb_solid_inner, nb_solid_outer);
    annulus_inner.NB_SOLID = nb_solid_inner;
    annulus_outer.NB_SOLID = nb_solid_outer;
    std::cout << "Bcsolid inner nodes: " << nb_solid_inner << std::endl;
    std::cout << "Bcsolid outer nodes: " << nb_solid_outer << std::endl;

#endif
}