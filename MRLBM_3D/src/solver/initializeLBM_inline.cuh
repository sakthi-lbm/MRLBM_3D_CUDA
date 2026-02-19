#ifndef INITIALIZE_LBM_INLINE_H
#define INITIALIZE_LBM_INLINE_H

#include "../all_headers.h"

inline void check_tau()
{
    if (TAU <= 0.5 || TAU >= 2.5)
    {
        printf("ERROR: Tau out of stable LBM range: tau = %f\n", TAU);
        exit(EXIT_FAILURE);
    }
}

inline void check_mach()
{
    const real Ma = U_MAX / sqrt(cs2); // cs2 = 1/3

    if (Ma > 0.33)
    {
        printf("ERROR: Mach number too high: Ma = %f\n", Ma);
        exit(EXIT_FAILURE);
    }
}

inline void initialize_host_device_constants()
{
    for (int q = 0; q < Q; q++)
    {
        h_Hxx[q] = h_cx[q] * h_cx[q] - cs2;
        h_Hyy[q] = h_cy[q] * h_cy[q] - cs2;
        h_Hzz[q] = h_cz[q] * h_cz[q] - cs2;
        h_Hxy[q] = h_cx[q] * h_cy[q];
        h_Hxz[q] = h_cx[q] * h_cz[q];
        h_Hyz[q] = h_cy[q] * h_cz[q];
    }
    // copy to GPU constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_w, h_w, sizeof(h_w)));
    checkCudaErrors(cudaMemcpyToSymbol(d_cx, h_cx, sizeof(h_cx)));
    checkCudaErrors(cudaMemcpyToSymbol(d_cy, h_cy, sizeof(h_cy)));
    checkCudaErrors(cudaMemcpyToSymbol(d_cz, h_cz, sizeof(h_cz)));

    checkCudaErrors(cudaMemcpyToSymbol(d_Hxx, h_Hxx, sizeof(h_Hxx)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hyy, h_Hyy, sizeof(h_Hyy)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hzz, h_Hzz, sizeof(h_Hzz)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hxy, h_Hxy, sizeof(h_Hxy)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hxz, h_Hxz, sizeof(h_Hxz)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hyz, h_Hyz, sizeof(h_Hyz)));

    check_tau();
    check_mach();
}

inline void initialize_nodeType(nodeVar &hMom)
{
    static_assert(NX % BLOCK_THREAD_X == 0, "NX must tile block size");
    static_assert(NY % BLOCK_THREAD_Y == 0, "NY must tile block size");
    static_assert(NZ % BLOCK_THREAD_Z == 0, "NZ must tile block size");
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
                hMom.nodeType[idx] = boundary_definitions(x, y, z);
            }
        }
    }
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

inline void write_geometry_files(nodeVar hMom)
{
    std::ofstream edges_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "edges.dat"), std::ios::trunc);
    std::ofstream corners_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "corners.dat"), std::ios::trunc);
    std::ofstream faces_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "faces.dat"), std::ios::trunc);
    std::ofstream fluid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "fluid.dat"), std::ios::trunc);

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
                else if (hMom.nodeType[idx] == BULK)
                {
                    fluid_file << x << " " << y << " " << z << " " << static_cast<int>(hMom.nodeType[idx]) << std::endl;
                }
            }
        }
    }

    // optional: close files (done automatically on destruction)
    faces_file.close();
    fluid_file.close();
}

#endif // INITIALIZE_LBM_INLINE_H
