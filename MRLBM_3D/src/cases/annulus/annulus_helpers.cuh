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

inline void buildBoundaryList_updateBoundaryNodeType(nodeVar &hMom, boundaryVar &h_annulus,
                                                     const nodeType_t BOUNDARY,
                                                     const nodeType_t BCFLUID_NODE,
                                                     const nodeType_t BCSOLID_NODE)
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

                if (hMom.nodeType[idx] >= BOUNDARY && hMom.nodeType[idx] < (BOUNDARY + 256))
                {
                    // std::cout << NB << " " << BOUNDARY << " " << count << " " << hMom.nodeType[idx] << std::endl;
                    if (count >= NB)
                    {
                        std::cout << "Overflow BOUNDARY at idx=" << idx << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    h_annulus.boundaryList[count] = idx;
                    count++;
                }
                else if (hMom.nodeType[idx] >= BCFLUID_NODE && hMom.nodeType[idx] < (BCFLUID_NODE + 256))
                {
                    // std::cout << NB << " " << BOUNDARY << " " << count << " " << hMom.nodeType[idx] << std::endl;
                    if (count2 >= NB_FLUID)
                    {
                        std::cout << "Overflow BCFLUID at idx=" << idx << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    h_annulus.bcfluidList[count2] = idx;
                    count2++;
                }
                else if (hMom.nodeType[idx] >= BCSOLID_NODE && hMom.nodeType[idx] < (BCSOLID_NODE + 256))
                {
                    // std::cout << NB << " " << BOUNDARY << " " << count << " " << hMom.nodeType[idx] << std::endl;
                    if (count3 >= NB_SOLID)
                    {
                        std::cout << "Overflow BCSOLID at idx=" << idx << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    h_annulus.bcsolidList[count3] = idx;
                    count3++;
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

inline void assignBoundaryIndices(nodeVar &hMom, boundaryVar &h_annulus, const nodeType_t BOUNDARY)
{

    // updating boundary nodetype with idx
    const int NB = h_annulus.NB;
    for (int i = 0; i < NB; i++)
    {
        const size_t idx = h_annulus.boundaryList[i];
        hMom.nodeType[idx] = BOUNDARY + i;
    }
}

inline void find_incomings_outgoings(const nodeVar &hMom,
                                     boundaryVar &annulus)
{
    const int nb = annulus.NB;
    if (nb <= 0)
        return;

    for (int i = 0; i < nb; i++)
    {
        uint32_t incomingMask = (1u << Q) - 1;
        uint32_t outgoingMask = 0;

        const size_t global_index = annulus.boundaryList[i];
        unsigned int x, y, z;
        GlobalIndexToXYZ(global_index, x, y, z);

        nodeType_t node[Q];
        load_neighbors(node, hMom, x, y, z);

        for (int q = 0; q < Q; q++)
        {
            // if any neighbour is solid the incoming from that node is 0
            if (node[q] == SOLID)
            {
                // clear incoming bit
                incomingMask &= ~(1u << opp[q]);
            }
        }

        for (int q = 0; q < Q; q++)
        {
            // outgoing is opposite of the incomings
            if (incomingMask & (1u << opp[q]))
            {
                outgoingMask |= (1u << q);
            }
        }
        annulus.incomingMask[i] = incomingMask;
        annulus.outgoingMask[i] = outgoingMask;
    }
}

inline void setup_bcfluid_masks(const nodeVar &hMom, boundaryVar &h_annulus, const nodeType_t BCFLUID_NODE)
{
    std::vector<bool> computed(MAX_NODE_TAG, false);

    const int NB_FLUID = h_annulus.NB_FLUID;

    for (int i = 0; i < NB_FLUID; i++)
    {
        size_t global_index = h_annulus.bcfluidList[i];
        unsigned int x, y, z;
        GlobalIndexToXYZ(global_index, x, y, z);

        int nodeTag = hMom.nodeType[global_index] - BCFLUID_NODE;

        if (global_index >= NX * NY * NZ)
        {
            std::cout << "ERROR: global_index out of range: " << global_index << std::endl;
            exit(1);
        }

        if (nodeTag < 0 || nodeTag >= 256)
        {
            std::cout << "ERROR: nodeTag out of range: " << nodeTag << std::endl;
            std::cout << i << " " << x << " " << y << " " << z << std::endl;
        }

        if (computed[nodeTag])
            continue;

        nodeType_t node[Q];
        load_neighbors(node, hMom, x, y, z);

        uint32_t incomingMask = (1u << Q) - 1;
        uint32_t outgoingMask = 0;

        for (int q = 0; q < Q; q++)
        {
            if (node[q] == SOLID)
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
    }
}

inline void setup_bcsolid_masks(const nodeVar &hMom, boundaryVar &h_annulus, const nodeType_t BCSOLID_NODE)
{
#if !Z_PERIODIC
    std::vector<bool> computed(MAX_NODE_TAG, false);

    const int NB_SOLID = h_annulus.NB_SOLID;

    for (int i = 0; i < NB_SOLID; i++)
    {
        size_t global_index = h_annulus.bcsolidList[i];

        int nodeTag = hMom.nodeType[global_index] - BCSOLID_NODE;

        if (computed[nodeTag])
            continue;

        unsigned int x, y, z;
        GlobalIndexToXYZ(global_index, x, y, z);

        nodeType_t node[Q];
        load_neighbors(node, hMom, x, y, z);

        uint32_t incomingMask = (1u << Q) - 1;
        uint32_t outgoingMask = 0;

        for (int q = 0; q < Q; q++)
        {
            if (node[q] == SOLID)
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
    return (t >= INNER_NODE && t <= INNER_NODE + 256);
}

inline bool isouter_cylinder(nodeType_t t)
{
    return (t >= OUTER_NODE && t <= OUTER_NODE + 256);
}

inline bool isBcfluid_inner(nodeType_t t)
{
    return (t >= BCFLUID_NODE_INNER && t <= BCFLUID_NODE_INNER + 256);
}

inline bool isBcfluid_outer(nodeType_t t)
{
    return (t >= BCFLUID_NODE_OUTER && t <= BCFLUID_NODE_OUTER + 256);
}

inline bool isBcsolid_inner(nodeType_t t)
{
    return (t >= BCSOLID_NODE_INNER && t <= BCSOLID_NODE_INNER + 256);
}

inline bool isBcsolid_outer(nodeType_t t)
{
    return (t >= BCSOLID_NODE_OUTER && t <= BCSOLID_NODE_OUTER + 256);
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

    const int Z_SLICE = -1;
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
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isouter_cylinder(hMom.nodeType[idx]))
                {
                    outer_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcfluid_inner(hMom.nodeType[idx]))
                {
                    bcfluid_inner_file << x << " " << y << " " << z << " "
                                       << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcfluid_outer(hMom.nodeType[idx]))
                {
                    bcfluid_outer_file << x << " " << y << " " << z << " "
                                       << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcsolid_inner(hMom.nodeType[idx]))
                {
                    bcsolid_inner_file << x << " " << y << " " << z << " "
                                       << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcsolid_outer(hMom.nodeType[idx]))
                {
                    bcsolid_outer_file << x << " " << y << " " << z << " "
                                       << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (hMom.nodeType[idx] == SOLID)
                {
                    solid_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (hMom.nodeType[idx] == BULK)
                {
                    fluid_file << x << " " << y << " " << z << " " << static_cast<int>(hMom.nodeType[idx]) << std::endl;
                }
                else
                {
                    others_file << x << " " << y << " " << z << " " << static_cast<int>(hMom.nodeType[idx]) << std::endl;
                }
            }
        }
    }
}
