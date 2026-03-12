#ifndef GEOMETRY_UTILS_CUH
#define GEOMETRY_UTILS_CUH

#include "file_utils.h"

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

    const int Z_SLICE = 1;
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

#endif // GEOMETRY_UTILS_CUH