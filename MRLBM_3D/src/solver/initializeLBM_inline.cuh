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

inline void initialize_cylinder_nodeType(nodeVar &hMom)
{
    for (int z = 0; z < NZ; z++)
    {
        for (int y = 0; y < NY; y++)
        {
            for (int x = 0; x < NX; x++)
            {
                // const real epsilon = 0.0;
                const real x_diff = toReal(x) - XC;
                const real y_diff = toReal(y) - YC;
                const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             z % BLOCK_THREAD_Z,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y,
                                             z / BLOCK_THREAD_Z);

                if (radius <= R_WALL)
                {
                    hMom.nodeType[idx] = SOLID;
                }
            }
        }
    }

    //================================ BOUNDARY NODES ========================================
    static_assert(LW >= 3 && LS >= 3);
    static_assert(LW + D + 3 < NX);
    static_assert(LS + D + 3 < NY);
    int count = 0;
    int node[Q];
    int xn, yn, zn;
    for (int z = 0; z < NZ; z++)
    {
        for (int y = (LS - 2); y < (LS + D + 2); y++)
        {
            for (int x = (LW - 2); x < (LW + D + 2); x++)
            {
                for (int q = 0; q < Q; q++)
                {
                    xn = x + h_cx[q];
                    yn = y + h_cy[q];
                    zn = z + h_cz[q];

#if Z_PERIODIC
                    if (zn < 0)
                        zn += NZ;
                    else if (zn >= NZ)
                        zn -= NZ;
                    node[q] = hMom.nodeType[IDX_BLOCK(xn % BLOCK_THREAD_X,
                                                      yn % BLOCK_THREAD_Y,
                                                      zn % BLOCK_THREAD_Z,
                                                      xn / BLOCK_THREAD_X,
                                                      yn / BLOCK_THREAD_Y,
                                                      zn / BLOCK_THREAD_Z)];
#else
                    if (zn < 0 || zn >= NZ)
                    {
                        node[q] = SOLID;
                    }

                    else
                    {
                        node[q] = hMom.nodeType[IDX_BLOCK(xn % BLOCK_THREAD_X,
                                                          yn % BLOCK_THREAD_Y,
                                                          zn % BLOCK_THREAD_Z,
                                                          xn / BLOCK_THREAD_X,
                                                          yn / BLOCK_THREAD_Y,
                                                          zn / BLOCK_THREAD_Z)];
                    }

#endif
                }

                // finding for any neighbour fluid
                bool anyFluid = false;
                for (int q = 1; q < Q; q++)
                {
                    if (node[q] == BULK)
                    {
                        anyFluid = true;
                        break;
                    }
                }

                // Process only boundary solid nodes
                if (node[0] == SOLID && anyFluid)
                {
                    const real x_diff = toReal(x) - XC;
                    const real y_diff = toReal(y) - YC;
                    const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

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

                    binary_t bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                    for (int b = 0; b < NUM_BITS; ++b)
                    {
                        for (int i = 0; i < DIRS_PER_BIT; ++i)
                        {
                            const int index = bit_dirs[b][i];

                            if (node[index] == BULK)
                            {
                                bits[b] = 1;
                                break;
                            }
                        }
                    }
                    const int node_tag = bits[0] * 1 + bits[1] * 2 + bits[2] * 4 + bits[3] * 8 +
                                         bits[4] * 16 + bits[5] * 32 + bits[6] * 64 + bits[7] * 128;

                    const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                                 y % BLOCK_THREAD_Y,
                                                 z % BLOCK_THREAD_Z,
                                                 x / BLOCK_THREAD_X,
                                                 y / BLOCK_THREAD_Y,
                                                 z / BLOCK_THREAD_Z);
                    if (z == 0)
                        count++;
                    hMom.nodeType[idx] = toNodeTypeT(INNER_NODE + node_tag);
                }
            }
        }
    }
    NB = count;
    std::cout << "inner: " << NB << std::endl;
}

inline void initialize_cylinder_nodeType_triangular(nodeVar &hMom)
{
    for (int z = 0; z < NZ; z++)
    {
        for (int y = 0; y < NY; y++)
        {
            for (int x = 0; x < NX; x++)
            {
                // const real epsilon = 0.0;
                const real x_diff = toReal(x) - XC;
                const real y_diff = toReal(y) - YC;
                const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             z % BLOCK_THREAD_Z,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y,
                                             z / BLOCK_THREAD_Z);

                if (radius <= R_WALL)
                {
                    hMom.nodeType[idx] = SOLID;
                }
            }
        }
    }

    //================================ BOUNDARY NODES ========================================
    static_assert(LW >= 3 && LS >= 3);
    static_assert(LW + D + 3 < NX);
    static_assert(LS + D + 3 < NY);
    int count = 0;
    int node[Q];
    int xn, yn, zn;
    for (int z = 0; z < NZ; z++)
    {
        for (int y = (LS - 2); y < (LS + D + 2); y++)
        {
            for (int x = (LW - 2); x < (LW + D + 2); x++)
            {
                for (int q = 0; q < Q; q++)
                {
                    xn = x + h_cx[q];
                    yn = y + h_cy[q];
                    zn = z + h_cz[q];

#if Z_PERIODIC
                    if (zn < 0)
                        zn += NZ;
                    else if (zn >= NZ)
                        zn -= NZ;
                    node[q] = hMom.nodeType[IDX_BLOCK(xn % BLOCK_THREAD_X,
                                                      yn % BLOCK_THREAD_Y,
                                                      zn % BLOCK_THREAD_Z,
                                                      xn / BLOCK_THREAD_X,
                                                      yn / BLOCK_THREAD_Y,
                                                      zn / BLOCK_THREAD_Z)];
#else
                    if (zn < 0 || zn >= NZ)
                    {
                        node[q] = SOLID;
                    }

                    else
                    {
                        node[q] = hMom.nodeType[IDX_BLOCK(xn % BLOCK_THREAD_X,
                                                          yn % BLOCK_THREAD_Y,
                                                          zn % BLOCK_THREAD_Z,
                                                          xn / BLOCK_THREAD_X,
                                                          yn / BLOCK_THREAD_Y,
                                                          zn / BLOCK_THREAD_Z)];
                    }

#endif
                }

                // finding for any neighbour fluid
                bool anyFluid = false;
                for (int q = 1; q < Q; q++)
                {
                    if (node[q] == BULK)
                    {
                        anyFluid = true;
                        break;
                    }
                }

                // Process only boundary solid nodes
                if (node[0] == SOLID && anyFluid)
                {
                    const real x_diff = toReal(x) - XC;
                    const real y_diff = toReal(y) - YC;
                    const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

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

                    binary_t bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                    for (int b = 0; b < NUM_BITS; ++b)
                    {
                        for (int i = 0; i < DIRS_PER_BIT; ++i)
                        {
                            const int index = bit_dirs[b][i];

                            if (node[index] == BULK)
                            {
                                bits[b] = 1;
                                break;
                            }
                        }
                    }
                    const int node_tag = bits[0] * 1 + bits[1] * 2 + bits[2] * 4 + bits[3] * 8 +
                                         bits[4] * 16 + bits[5] * 32 + bits[6] * 64 + bits[7] * 128;

                    const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                                 y % BLOCK_THREAD_Y,
                                                 z % BLOCK_THREAD_Z,
                                                 x / BLOCK_THREAD_X,
                                                 y / BLOCK_THREAD_Y,
                                                 z / BLOCK_THREAD_Z);
                    if (z == 0)
                        count++;
                    hMom.nodeType[idx] = toNodeTypeT(INNER_NODE + node_tag);
                }
            }
        }
    }
    NB = count;
    std::cout << "inner: " << NB << std::endl;
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

inline void write_geometry_files(nodeVar hMom)
{
    std::ofstream edges_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "edges.dat"), std::ios::trunc);
    std::ofstream corners_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "corners.dat"), std::ios::trunc);
    std::ofstream faces_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "faces.dat"), std::ios::trunc);
    std::ofstream fluid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "fluid.dat"), std::ios::trunc);
    std::ofstream bound_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bound.dat"), std::ios::trunc);
    std::ofstream solid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "solid.dat"), std::ios::trunc);

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

                if (isFace(hMom.nodeType[idx]) && z == 0)
                {
                    faces_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isEdge(hMom.nodeType[idx]) && z == 0)
                {
                    edges_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isCorner(hMom.nodeType[idx]) && z == 0)
                {
                    corners_file << x << " " << y << " " << z << " "
                                 << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (isCylinder(hMom.nodeType[idx]) && z == 0)
                {
                    bound_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (hMom.nodeType[idx] == SOLID && z == 0)
                {
                    solid_file << x << " " << y << " " << z << " "
                               << static_cast<int>(hMom.nodeType[idx]) << "\n";
                }
                else if (hMom.nodeType[idx] == BULK && z == 0)
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
