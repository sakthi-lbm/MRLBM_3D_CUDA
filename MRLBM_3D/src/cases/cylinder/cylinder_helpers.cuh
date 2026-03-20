#ifndef INITIALIZE_LBM_INLINE_H
#define INITIALIZE_LBM_INLINE_H

#include <unordered_map>
#include <vector>

#include "../../boundary/curvedLBM.cuh"

inline void buildBoundaryList_updateBoundaryNodeType(nodeVar &hMom, cylinderVar &h_cylinder)
{
    const int NB = h_cylinder.NB;
    const int NB_FLUID = h_cylinder.NB_FLUID;
    const int NB_SOLID = h_cylinder.NB_SOLID;

    int count = 0;
    int count2 = 0;
    int count3 = 0;
    for (int z = 0; z < NZ; z++)
    {
        for (int y = (LS - 2); y < (LS + D + 2); y++)
        {
            for (int x = (LW - 2); x < (LW + D + 2); x++)
            {
                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             z % BLOCK_THREAD_Z,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y,
                                             z / BLOCK_THREAD_Z);

                if (hMom.nodeType[idx] >= INNER_NODE && hMom.nodeType[idx] < (INNER_NODE + 256))
                {
                    if (count >= NB)
                    {
                        printf("ERROR: boundaryList overflow\n");
                        exit(EXIT_FAILURE);
                    }
                    h_cylinder.boundaryList[count] = idx;
                    count++;
                }
                else if (hMom.nodeType[idx] >= BCFLUID_NODE && hMom.nodeType[idx] < (BCFLUID_NODE + 256))
                {
                    if (count2 >= NB_FLUID)
                    {
                        printf("ERROR: BcfluidList overflow\n");
                        exit(EXIT_FAILURE);
                    }
                    h_cylinder.bcfluidList[count2] = idx;
                    count2++;
                }
                else if (hMom.nodeType[idx] >= BCSOLID_NODE && hMom.nodeType[idx] < (BCSOLID_NODE + 256))
                {
                    if (count3 >= NB_SOLID)
                    {
                        printf("ERROR: BcsolidList overflow\n");
                        exit(EXIT_FAILURE);
                    }
                    h_cylinder.bcsolidList[count3] = idx;
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

    // updating boundary nodetype with idx
    for (int i = 0; i < NB; i++)
    {
        const size_t idx = h_cylinder.boundaryList[i];
        hMom.nodeType[idx] = toNodeTypeT(INNER_NODE) + i;
    }
}

inline void find_incomings_outgoings(const nodeVar &hMom,
                                     cylinderVar &cylinder)
{
    const int nb = cylinder.NB;
    if (nb <= 0)
        return;

    for (int i = 0; i < nb; i++)
    {
        uint32_t incomingMask = (1u << Q) - 1;
        uint32_t outgoingMask = 0;

        const size_t global_index = cylinder.boundaryList[i];
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
        cylinder.incomingMask[i] = incomingMask;
        cylinder.outgoingMask[i] = outgoingMask;

        // // debuggig
        // if (z == 0)
        // {
        //     std::cout << "node " << i << " at (x, y,z) = (" << x << ", " << y << ")\n";
        //     for (int q = 0; q < Q; q++)
        //     {
        //         binary_t incomingMaskBit = (incomingMask >> q) & 1u;
        //         binary_t outgoingMaskBit = (outgoingMask >> q) & 1u;

        //         std::cout << " q=" << q << " nodetag=" << node[0]
        //                   << " incomingMask=" << static_cast<int>(incomingMaskBit)
        //                   << " outgoingMask=" << static_cast<int>(outgoingMaskBit)
        //                   << "\n";
        //     }
        // }
    }
}

inline void setup_bcfluid_masks(const nodeVar &hMom, cylinderVar &h_cylinder)
{
    std::vector<bool> computed(MAX_NODE_TAG, false);

    const int NB_FLUID = h_cylinder.NB_FLUID;

    for (int i = 0; i < NB_FLUID; i++)
    {
        size_t global_index = h_cylinder.bcfluidList[i];

        int nodeTag = hMom.nodeType[global_index] - BCFLUID_NODE;

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

        h_incomingMask_bcfluid[nodeTag] = incomingMask;
        h_outgoingMask_bcfluid[nodeTag] = outgoingMask;

        computed[nodeTag] = true;
    }
}

inline void setup_bcsolid_masks(const nodeVar &hMom, cylinderVar &h_cylinder)
{
#if !Z_PERIODIC
    std::vector<bool> computed(MAX_NODE_TAG, false);

    const int NB_SOLID = h_cylinder.NB_SOLID;

    for (int i = 0; i < NB_SOLID; i++)
    {
        size_t global_index = h_cylinder.bcsolidList[i];

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

inline bool isEdge(nodeType_t t)
{
    return (t == NORTH_WEST || t == SOUTH_WEST ||
            t == WEST_FRONT || t == WEST_BACK ||
            t == NORTH_EAST || t == SOUTH_EAST ||
            t == EAST_FRONT || t == EAST_BACK ||
            t == NORTH_FRONT || t == NORTH_BACK ||
            t == SOUTH_FRONT || t == SOUTH_BACK);
}

inline bool isFace(nodeType_t t)
{
    return (t == NORTH || t == SOUTH ||
            t == WEST || t == EAST ||
            t == FRONT || t == BACK);
}

inline bool isCorner(nodeType_t t)
{
    return (t == NORTH_WEST_FRONT || t == NORTH_WEST_BACK ||
            t == SOUTH_WEST_FRONT || t == SOUTH_WEST_BACK ||
            t == NORTH_EAST_FRONT || t == NORTH_EAST_BACK ||
            t == SOUTH_EAST_FRONT || t == SOUTH_EAST_BACK);
}

inline bool isCylinder(nodeType_t t)
{
    return (t >= INNER_NODE && t <= INNER_NODE + 256);
}

inline bool isBcfluid(nodeType_t t)
{
    return (t >= BCFLUID_NODE && t <= BCFLUID_NODE + 256);
}

inline bool isBcsolid(nodeType_t t)
{
    return (t >= BCSOLID_NODE && t <= BCSOLID_NODE + 256);
}

inline void write_geometry_files(nodeVar hMom)
{
    std::ofstream edges_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "edges.dat"), std::ios::trunc);
    std::ofstream corners_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "corners.dat"), std::ios::trunc);
    std::ofstream faces_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "faces.dat"), std::ios::trunc);
    std::ofstream fluid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "fluid.dat"), std::ios::trunc);
    std::ofstream bound_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bound.dat"), std::ios::trunc);
    std::ofstream bcfluid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bcfluid.dat"), std::ios::trunc);
    std::ofstream bcsolid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bcsolid.dat"), std::ios::trunc);
    std::ofstream solid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "solid.dat"), std::ios::trunc);
    std::ofstream others_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "others.dat"), std::ios::trunc);

    const int Z_SLICE = NZ - 1;
    for (int z = 0; z < NZ; z++)
    {
        if (Z_SLICE >= 0 && z != Z_SLICE)
            continue;

        // for (int y = 0; y < NY; y++)
        for (int y = (LS - 2); y < (LS + D + 2); y++)
        {
            // for (int x = 0; x < NX; x++)
            for (int x = (LW - 2); x < (LW + D + 2); x++)
            {

                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             z % BLOCK_THREAD_Z,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y,
                                             z / BLOCK_THREAD_Z);

                if (isFace(hMom.nodeType[idx]))
                {
                    faces_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isEdge(hMom.nodeType[idx]))
                {
                    edges_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isCorner(hMom.nodeType[idx]))
                {
                    corners_file << x << " " << y << " " << z << " "
                                 << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isCylinder(hMom.nodeType[idx]))
                {
                    bound_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcfluid(hMom.nodeType[idx]))
                {
                    bcfluid_file << x << " " << y << " " << z << " "
                                 << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isBcsolid(hMom.nodeType[idx]))
                {
                    bcsolid_file << x << " " << y << " " << z << " "
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

#endif // INITIALIZE_LBM_INLINE_H
