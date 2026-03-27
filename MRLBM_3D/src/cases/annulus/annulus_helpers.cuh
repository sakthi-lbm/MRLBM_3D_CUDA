#pragma once

#include <unordered_map>
#include <vector>

#include "nodeClass.cuh"

inline void compute_unit_vectors_boundary_nodes(nodeVar &hMom, boundaryVar &h_annulus, const real D_wall)
{
    const int nb = h_annulus.NB;
    if (nb <= 0)
        return;

    for (int i = 0; i < nb; i++)
    {
        const size_t global_index = h_annulus.boundaryList[i];
        unsigned int x, y, z;
        GlobalIndexToXYZ(global_index, x, y, z);

        // Boundary node location
        const real xb = toReal(x);
        const real yb = toReal(y);
        const real zb = toReal(z);

        // unit normal calculation
        const real dx = xb - XC;
        const real dy = yb - YC;
        const real radius2 = dx * dx + dy * dy;
        const real inv_radius = rsqrt(radius2);

        const real unit_nx = dx * inv_radius;
        const real unit_ny = dy * inv_radius;

        // wall point location (annulus)
        const real r_wall = toReal(0.5) * D_wall;
        const real xw = XC + r_wall * unit_nx;
        const real yw = YC + r_wall * unit_ny;

        const real delta = rabs((xw - xb) * unit_nx + (yw - yb) * unit_ny);

        h_annulus.unit_nx[i] = unit_nx;
        h_annulus.unit_ny[i] = unit_ny;
        h_annulus.delta_w[i] = delta;
    }
}

inline void buildBoundaryList_updateBoundaryNodeType(nodeVar &hMom,
                                                     boundaryVar &h_annulus,
                                                     const uint32_t NODE_BOUNDARY_TYPE,
                                                     const uint32_t NODE_BCFLUID_TYPE,
                                                     const uint32_t NODE_BCSOLID_TYPE)
{
    const int NB = h_annulus.NB;
    const int NB_FLUID = h_annulus.NB_FLUID;
    const int NB_SOLID = h_annulus.NB_SOLID;

    int count = 0;
    int count2 = 0;
    int count3 = 0;

    for (int z = 0; z < NZ; z++)
    {
        for (int y = 0; y < NY; y++)
        {
            for (int x = 0; x < NX; x++)
            {
                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             z % BLOCK_THREAD_Z,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y,
                                             z / BLOCK_THREAD_Z);

                const uint32_t type = getType(hMom.nodeType[idx]);

                if (type == NODE_BOUNDARY_TYPE)
                {
                    if (count >= NB)
                    {
                        std::cout << "Overflow BOUNDARY at idx=" << idx << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    h_annulus.boundaryList[count++] = idx;
                }
                else if (type == NODE_BCFLUID_TYPE)
                {
                    if (count2 >= NB_FLUID)
                    {
                        std::cout << "Overflow BCFLUID at idx=" << idx << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    h_annulus.bcfluidList[count2++] = idx;
                }
                else if (type == NODE_BCSOLID_TYPE)
                {
                    if (count3 >= NB_SOLID)
                    {
                        std::cout << "Overflow BCSOLID at idx=" << idx << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    h_annulus.bcsolidList[count3++] = idx;
                }
            }
        }
    }

    if (count != NB)
    {
        printf("ERROR: Boundary count mismatch! count=%d NB=%d\n", count, NB);
        exit(EXIT_FAILURE);
    }
}

inline void assignBoundaryIndices(nodeVar &hMom, boundaryVar &h_annulus, uint32_t type)
{

    // updating boundary nodetype with idx
    const int NB = h_annulus.NB;
    for (int i = 0; i < NB; i++)
    {
        if (i >= (1 << TYPE_SHIFT))
        {
            std::cout << "Index overflow in encodeNode!" << std::endl;
            exit(EXIT_FAILURE);
        }

        const size_t idx = h_annulus.boundaryList[i];
        hMom.nodeType[idx] = encodeNode(type, i);
    }
}

inline void find_incomings_outgoings(nodeVar &hMom, boundaryVar &annulus)
{
    const int nb = annulus.NB;
    if (nb <= 0)
        return;

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

    for (int i = 0; i < nb; i++)
    {
        const size_t global_index = annulus.boundaryList[i];

        unsigned int x, y, z;
        GlobalIndexToXYZ(global_index, x, y, z);

        nodeType_t node[Q];
        load_neighbors(node, hMom, x, y, z);

        binary_t bits[8];

        // Step 1: compute valid bits
        for (int b = 0; b < 8; b++)
        {
            bits[b] = 1;

            for (int k = 0; k < 7; k++)
            {
                int q = bit_dirs[b][k];

                if (getType(node[q]) == NODE_SOLID)
                {
                    bits[b] = 0;
                    break;
                }
            }
        }

        // Step 2: build incomingMask
        uint32_t incomingMask = 0;
        incomingMask |= (1u << 0);

        for (int b = 0; b < 8; b++)
        {
            if (!bits[b])
                continue;

            for (int k = 0; k < 7; k++)
            {
                int q = bit_dirs[b][k];
                incomingMask |= (1u << opp[q]);
            }
        }

        // Step 3: outgoingMask
        uint32_t outgoingMask = 0;
        outgoingMask |= (1u << 0);

        for (int q = 0; q < Q; q++)
        {
            if (incomingMask & (1u << opp[q]))
                outgoingMask |= (1u << q);
        }

        annulus.incomingMask[i] = incomingMask;
        annulus.outgoingMask[i] = outgoingMask;

        // if (z == 1)
        // {
        //     std::cout << "node " << i << " at (x, y,z) = (" << x << ", " << y << ")\n";
        //     for (int q = 0; q < Q; q++)
        //     {
        //         binary_t incomingMaskBit = (incomingMask >> q) & 1u;
        //         binary_t outgoingMaskBit = (outgoingMask >> q) & 1u;

        //         std::cout << " q=" << q
        //                   << " type=" << getType(node[q])
        //                   << " tag=" << getIndex(node[q])
        //                   << " incomingMask=" << static_cast<int>(incomingMaskBit)
        //                   << " outgoingMask=" << static_cast<int>(outgoingMaskBit)
        //                   << "\n";
        //     }
        // }
    }
}

inline void setup_bcfluid_masks(const nodeVar &hMom, boundaryVar &h_annulus)
{
    std::vector<bool> computed(MAX_NODE_TAG, false);

    const int NB_FLUID = h_annulus.NB_FLUID;

    for (int i = 0; i < NB_FLUID; i++)
    {
        size_t global_index = h_annulus.bcfluidList[i];
        unsigned int x, y, z;
        GlobalIndexToXYZ(global_index, x, y, z);

        nodeType_t nodeType = hMom.nodeType[global_index];

        if (getType(nodeType) != NODE_BCFLUID_INNER &&
            getType(nodeType) != NODE_BCFLUID_OUTER)
        {
            printf("ERROR: Wrong node type in bcfluidList\n");
            exit(1);
        }

        nodeType_t nodeTag = getIndex(nodeType);

        if (nodeTag >= MAX_NODE_TAG)
        {
            printf("ERROR: nodeTag out of range: %u\n", nodeTag);
            exit(1);
        }

        if (computed[nodeTag])
            continue;

        nodeType_t node[Q];
        load_neighbors(node, hMom, x, y, z);

        uint32_t incomingMask = (1u << Q) - 1;
        uint32_t outgoingMask = 0;

        for (int q = 0; q < Q; q++)
        {
            if (getType(node[q]) == NODE_SOLID)
                incomingMask &= ~(1u << opp[q]);
        }

        for (int q = 0; q < Q; q++)
        {
            if (incomingMask & (1u << opp[q]))
                outgoingMask |= (1u << q);
        }

        h_incomingMask_bcfluid[nodeTag] = incomingMask;
        h_outgoingMask_bcfluid[nodeTag] = outgoingMask;

        computed[nodeTag] = true;

        // if (z == 1)
        // {
            std::cout << "node " << i << " at (x, y,z) = (" << x << ", " << y << ")\n";
            for (int q = 0; q < Q; q++)
            {
                binary_t incomingMaskBit = (incomingMask >> q) & 1u;
                binary_t outgoingMaskBit = (outgoingMask >> q) & 1u;

                std::cout << " q=" << q
                          << " type=" << getType(node[q])
                          << " tag=" << getIndex(node[q])
                          << " incomingMask=" << static_cast<int>(incomingMaskBit)
                          << " outgoingMask=" << static_cast<int>(outgoingMaskBit)
                          << "\n";
            }
        // }
    }
}

inline void setup_bcsolid_masks(const nodeVar &hMom, boundaryVar &h_annulus)
{
#if !Z_PERIODIC
    std::vector<bool> computed(MAX_NODE_TAG, false);

    const int NB_SOLID = h_annulus.NB_SOLID;

    for (int i = 0; i < NB_SOLID; i++)
    {
        size_t global_index = h_annulus.bcsolidList[i];
        unsigned int x, y, z;
        GlobalIndexToXYZ(global_index, x, y, z);

        nodeType_t nodeType = hMom.nodeType[global_index];

        if (getType(nodeType) != NODE_BCSOLID_INNER &&
            getType(nodeType) != NODE_BCSOLID_OUTER)
        {
            printf("ERROR: Wrong node type in bcsolidList\n");
            exit(1);
        }

        nodeType_t nodeTag = getIndex(nodeType);
        if (nodeTag >= MAX_NODE_TAG)
        {
            printf("ERROR: nodeTag out of range: %u\n", nodeTag);
            exit(1);
        }

        if (computed[nodeTag])
            continue;

        nodeType_t node[Q];
        load_neighbors(node, hMom, x, y, z);

        uint32_t incomingMask = (1u << Q) - 1;
        uint32_t outgoingMask = 0;

        for (int q = 0; q < Q; q++)
        {
            if (getType(node[q]) == NODE_SOLID)
                incomingMask &= ~(1u << opp[q]);
        }

        for (int q = 0; q < Q; q++)
        {
            if (incomingMask & (1u << opp[q]))
                outgoingMask |= (1u << q);
        }

        h_incomingMask_bcsolid[nodeTag] = incomingMask;
        h_outgoingMask_bcsolid[nodeTag] = outgoingMask;

        computed[nodeTag] = true;
    }
#endif
}

inline bool isinner_cylinder(nodeType_t t)
{
    return getType(t) == NODE_INNER;
}

inline bool isouter_cylinder(nodeType_t t)
{
    return getType(t) == NODE_OUTER;
}

inline bool isBcfluid_inner(nodeType_t t)
{
    return getType(t) == NODE_BCFLUID_INNER;
}

inline bool isBcfluid_outer(nodeType_t t)
{
    return getType(t) == NODE_BCFLUID_OUTER;
}

inline bool isBcsolid_inner(nodeType_t t)
{
    return getType(t) == NODE_BCSOLID_INNER;
}

inline bool isBcsolid_outer(nodeType_t t)
{
    return getType(t) == NODE_BCSOLID_OUTER;
}

inline void write_geometry_files(nodeVar hMom)
{
    std::ofstream fluid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "fluid.dat"), std::ios::trunc);
    std::ofstream solid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "solid.dat"), std::ios::trunc);
    std::ofstream others_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "others.dat"), std::ios::trunc);

    std::ofstream inner_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "inner.dat"), std::ios::trunc);
    std::ofstream outer_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "outer.dat"), std::ios::trunc);

    std::ofstream bcfluid_inner_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bcfluid_inner.dat"), std::ios::trunc);
    std::ofstream bcfluid_outer_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bcfluid_outer.dat"), std::ios::trunc);

    std::ofstream bcsolid_inner_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bcsolid_inner.dat"), std::ios::trunc);
    std::ofstream bcsolid_outer_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bcsolid_outer.dat"), std::ios::trunc);

    const int Z_SLICE = 10;
    for (int z = 0; z < NZ; z++)
    {
        if (Z_SLICE >= 0 && z != Z_SLICE)
            continue;

        // for (int y = 0; y < NY; y++)
        for (int y = 0; y < NY; y++)
        {
            // for (int x = 0; x < NX; x++)
            for (int x = 0; x < NX; x++)
            {
                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             z % BLOCK_THREAD_Z,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y,
                                             z / BLOCK_THREAD_Z);

                if (isinner_cylinder(hMom.nodeType[idx]))
                {
                    inner_file << x << " " << y << " " << z << " "
                               << getType(hMom.nodeType[idx]) << "\n";
                }
                else if (isouter_cylinder(hMom.nodeType[idx]))
                {
                    outer_file << x << " " << y << " " << z << " " << getType(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcfluid_inner(hMom.nodeType[idx]))
                {
                    bcfluid_inner_file << x << " " << y << " " << z << " " << getType(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcfluid_outer(hMom.nodeType[idx]))
                {
                    bcfluid_outer_file << x << " " << y << " " << z << " " << getType(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcsolid_inner(hMom.nodeType[idx]))
                {
                    bcsolid_inner_file << x << " " << y << " " << z << " " << getType(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcsolid_outer(hMom.nodeType[idx]))
                {
                    bcsolid_outer_file << x << " " << y << " " << z << " " << getType(hMom.nodeType[idx]) << "\n";
                }
                else if (getType(hMom.nodeType[idx]) == NODE_SOLID)
                {
                    solid_file << x << " " << y << " " << z << " " << getType(hMom.nodeType[idx]) << "\n";
                }
                else if (getType(hMom.nodeType[idx]) == NODE_BULK)
                {
                    fluid_file << x << " " << y << " " << z << " " << getType(hMom.nodeType[idx]) << std::endl;
                }
                else
                {
                    others_file << x << " " << y << " " << z << " " << getType(hMom.nodeType[idx]) << std::endl;
                }
            }
        }
    }
}
