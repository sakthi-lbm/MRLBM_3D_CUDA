#ifndef INITIALIZE_LBM_INLINE_H
#define INITIALIZE_LBM_INLINE_H

#include "../all_headers.h"
#include "../globalStructs.h"
#include "../halo_interface/halo_interface.cuh"

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
    real Hxx[Q];
    real Hyy[Q];
    real Hxy[Q];
    for (size_t q = 0; q < Q; q++)
    {
        Hxx[q] = h_cx[q] * h_cx[q] - cs2;
        Hyy[q] = h_cy[q] * h_cy[q] - cs2;
        Hxy[q] = h_cx[q] * h_cy[q];
    }
    // copy to GPU constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_Hxx, &Hxx, sizeof(Hxx)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hyy, &Hyy, sizeof(Hyy)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hxy, &Hxy, sizeof(Hxy)));

    check_tau();
    check_mach();

    checkCudaErrors(cudaMemcpyToSymbol(d_NB, &NB, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_NBCF, &NBCF, sizeof(int)));
}

inline void initialize_nodeType(nodeVar hMom)
{
    for (int y = 0; y < NY; y++)
    {
        for (int x = 0; x < NX; x++)
        {
            const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                         y % BLOCK_THREAD_Y,
                                         x / BLOCK_THREAD_X,
                                         y / BLOCK_THREAD_Y);
            hMom.nodeType[idx] = boundary_definitions(x, y);
        }
    }
}

inline void initialize_cylinder_nodeType(nodeVar hMom)
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
                                         x / BLOCK_THREAD_X,
                                         y / BLOCK_THREAD_Y);

            if (radius <= R_WALL)
            {
                hMom.nodeType[idx] = SOLID;
            }
        }
    }

    //================================ BOUNDARY NODES ========================================
    int count = 0;
    int node[Q];

    for (int y = (L_BOT - 2); y < (L_BOT + D + 2); y++)
    {
        for (int x = (L_UP - 2); x < (L_UP + D + 2); x++)
        {
            for (int q = 0; q < Q; q++)
            {
                const int xn = x + h_cx[q];
                const int yn = y + h_cy[q];

                node[q] = hMom.nodeType[IDX_BLOCK(xn % BLOCK_THREAD_X,
                                                  yn % BLOCK_THREAD_Y,
                                                  xn / BLOCK_THREAD_X,
                                                  yn / BLOCK_THREAD_Y)];
            }
            const bool anyFluid = (node[1] == BULK ||
                                   node[2] == BULK ||
                                   node[3] == BULK ||
                                   node[4] == BULK ||
                                   node[5] == BULK ||
                                   node[6] == BULK ||
                                   node[7] == BULK ||
                                   node[8] == BULK);

            // Process only boundary solid nodes
            if (node[0] == SOLID && anyFluid)
            {
                const real x_diff = toReal(x) - XC;
                const real y_diff = toReal(y) - YC;
                const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

                // Calculate bits for moment reconstruction stencil deficiency
                const int bit_1 = (node[3] != BULK && node[4] != BULK && node[7] != BULK) ? 0 : 1;
                const int bit_2 = (node[1] != BULK && node[4] != BULK && node[8] != BULK) ? 0 : 1;
                const int bit_4 = (node[2] != BULK && node[3] != BULK && node[6] != BULK) ? 0 : 1;
                const int bit_8 = (node[1] != BULK && node[2] != BULK && node[5] != BULK) ? 0 : 1;
                const int node_tag = bit_1 * 1 + bit_2 * 2 + bit_4 * 4 + bit_8 * 8;

                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y);

                count++;
                hMom.nodeType[idx] = toNodeTypeT(INNER_NODE + node_tag);
                {
                }
            }
        }
        NB = count;
    }

    std::cout << "inner: " << NB << std::endl;
}

inline void initialize_cylinder_nodeType_triangular(nodeVar hMom)
{
    for (int y = (L_BOT - 2); y < (L_BOT + D + 2); y++)
    {
        for (int x = (L_UP - 2); x < (L_UP + D + 2); x++)
        {
            const real x_diff = toReal(x) - XC;
            const real y_diff = toReal(y) - YC;
            const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

            const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                         y % BLOCK_THREAD_Y,
                                         x / BLOCK_THREAD_X,
                                         y / BLOCK_THREAD_Y);

            if (radius <= R_WALL)
            {
                hMom.nodeType[idx] = SOLID;
            }
        }
    }
    //================================ BOUNDARY NODES ========================================
    int count = 0;
    int node[Q];

    for (int y = (L_BOT - 2); y < (L_BOT + D + 2); y++)
    {
        for (int x = (L_UP - 2); x < (L_UP + D + 2); x++)
        {
            for (int q = 0; q < Q; q++)
            {
                const int xn = x + h_cx[q];
                const int yn = y + h_cy[q];

                node[q] = hMom.nodeType[IDX_BLOCK(
                    xn % BLOCK_THREAD_X,
                    yn % BLOCK_THREAD_Y,
                    xn / BLOCK_THREAD_X,
                    yn / BLOCK_THREAD_Y)];
            }
            const bool anyFluid = (node[1] == BULK ||
                                   node[2] == BULK ||
                                   node[3] == BULK ||
                                   node[4] == BULK ||
                                   node[5] == BULK ||
                                   node[6] == BULK ||
                                   node[7] == BULK ||
                                   node[8] == BULK);

            // Process only boundary solid nodes
            if (node[0] == SOLID && anyFluid)
            {
                // Calculate bits for moment reconstruction stencil deficiency
                const int bit_1 = (node[3] != BULK && node[4] != BULK && node[7] != BULK) ? 0 : 1;
                const int bit_2 = (node[1] != BULK && node[4] != BULK && node[8] != BULK) ? 0 : 1;
                const int bit_4 = (node[2] != BULK && node[3] != BULK && node[6] != BULK) ? 0 : 1;
                const int bit_8 = (node[1] != BULK && node[2] != BULK && node[5] != BULK) ? 0 : 1;

                if ((bit_1 + bit_2 + bit_4 + bit_8) > 1)
                {
                    const real x_diff = toReal(x) - XC;
                    const real y_diff = toReal(y) - YC;
                    const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

                    const int node_tag = bit_1 * 1 + bit_2 * 2 + bit_4 * 4 + bit_8 * 8;

                    const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y);

                    count++;
                    hMom.nodeType[idx] = toNodeTypeT(INNER_NODE + node_tag);
                }
            }
        }
    }
    NB = count;

    std::cout << "inner: " << NB << std::endl;

    //=============================== BC FLUID NODES ==========================================
    int bc_count = 0;
    for (int y = (L_BOT - 2); y < (L_BOT + D + 2); y++)
    {
        for (int x = (L_UP - 2); x < (L_UP + D + 2); x++)
        {
            for (int q = 0; q < Q; q++)
            {
                const int xn = x + h_cx[q];
                const int yn = y + h_cy[q];

                node[q] = hMom.nodeType[IDX_BLOCK(xn % BLOCK_THREAD_X,
                                                  yn % BLOCK_THREAD_Y,
                                                  xn / BLOCK_THREAD_X,
                                                  yn / BLOCK_THREAD_Y)];
            }
            const bool anySolid = node[1] == SOLID ||
                                  node[2] == SOLID ||
                                  node[3] == SOLID ||
                                  node[4] == SOLID ||
                                  node[5] == SOLID ||
                                  node[6] == SOLID ||
                                  node[7] == SOLID ||
                                  node[8] == SOLID;

            if (node[0] == BULK && anySolid)
            {
                const real x_diff = toReal(x) - XC;
                const real y_diff = toReal(y) - YC;
                const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);

                // finding tag for the bcfluid nodes
                const int bit_1 = node[3] != SOLID && node[4] != SOLID && node[7] != SOLID ? 0 : 1;
                const int bit_2 = node[1] != SOLID && node[4] != SOLID && node[8] != SOLID ? 0 : 1;
                const int bit_4 = node[2] != SOLID && node[3] != SOLID && node[6] != SOLID ? 0 : 1;
                const int bit_8 = node[1] != SOLID && node[2] != SOLID && node[5] != SOLID ? 0 : 1;

                const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                             y % BLOCK_THREAD_Y,
                                             x / BLOCK_THREAD_X,
                                             y / BLOCK_THREAD_Y);

                bc_count++;
                if (bit_1)
                {
                    hMom.nodeType[idx] = toNodeTypeT(BCFLUID_NODE + 0);
                }
                else if (bit_2)
                {
                    hMom.nodeType[idx] = toNodeTypeT(BCFLUID_NODE + 1);
                }
                else if (bit_4)
                {
                    hMom.nodeType[idx] = toNodeTypeT(BCFLUID_NODE + 2);
                }
                else if (bit_8)
                {
                    hMom.nodeType[idx] = toNodeTypeT(BCFLUID_NODE + 3);
                }
            }
        }
    }
    NBCF = bc_count;

    std::cout << "BCF_inner: " << NBCF << std::endl;

    // int solid_count = 0, bulk_count = 0,
    //     boundary_count_inner = 0, boundary_count_outer = 0, bcf_count_inner = 0, bcf_count_outer = 0;
    // for (int y = 0; y < NY; y++)
    // {
    //     for (int x = 0; x < NX; x++)
    //     {
    //         const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y,
    //                                      x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y);
    //         nodeType_t type = hMom.nodeType[idx];

    //         if (type == SOLID)
    //             solid_count++;
    //         else if (type == BULK)
    //             bulk_count++;
    //         else if (type >= INNER_NODE && type < INNER_NODE + 16)
    //             boundary_count_inner++;
    //         else if (type >= OUTER_NODE && type < OUTER_NODE + 16)
    //             boundary_count_outer++;
    //         else if (type >= BCFLUID_INNER_NODE && type < BCFLUID_INNER_NODE + 4)
    //             bcf_count_inner++;
    //         else if (type >= BCFLUID_OUTER_NODE && type < BCFLUID_OUTER_NODE + 4)
    //             bcf_count_outer++;
    //     }
    // }

    // std::cout << "Domain Statistics:" << std::endl;
    // std::cout << "  SOLID nodes: " << solid_count << std::endl;
    // std::cout << "  BULK nodes: " << bulk_count << std::endl;
    // std::cout << "  INNER BOUNDARY nodes: " << boundary_count_inner << std::endl;
    // std::cout << "  OUTER BOUNDARY nodes: " << boundary_count_outer << std::endl;
    // std::cout << "  INNER BCFLUID nodes: " << bcf_count_inner << std::endl;
    // std::cout << "  OUTER BCFLUID nodes: " << bcf_count_outer << std::endl;
    // std::cout << "  Total: " << (solid_count + bulk_count + boundary_count_inner + boundary_count_outer + bcf_count_inner + bcf_count_outer)
    //           << " (expected: " << NX * NY << ")" << std::endl;
}

inline void buildBoundaryList_updateBoundaryNodeType(nodeVar &fMom, cylinderVar &h_cylinder)
{
    int count = 0;
    for (int y = (L_BOT - 2); y < (L_BOT + D + 2); y++)
    {
        for (int x = (L_UP - 2); x < (L_UP + D + 2); x++)
        {
            const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                         y % BLOCK_THREAD_Y,
                                         x / BLOCK_THREAD_X,
                                         y / BLOCK_THREAD_Y);

            if (fMom.nodeType[idx] >= INNER_NODE && fMom.nodeType[idx] < (INNER_NODE + 16))
            {
                if (count >= NB)
                {
                    printf("ERROR: boundaryList overflow\n");
                    exit(EXIT_FAILURE);
                }
                h_cylinder.boundaryList[count] = idx;
                count++;
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
        fMom.nodeType[idx] = toNodeTypeT(INNER_NODE + i);
    }
}

inline void find_incomings_outgoings_cylinder(const nodeVar &hMom, cylinderVar &h_cylinder, const int nb)
{
    if (nb <= 0)
        return;
    const size_t nbytes = toSize_t(nb) * toSize_t(Q) * sizeof(binary_t);

    memset(h_cylinder.incomings, 1, nbytes);
    memset(h_cylinder.outgoings, 0, nbytes);

    for (int i = 0; i < nb; i++)
    {
        const size_t global_index = h_cylinder.boundaryList[i];
        unsigned int x, y;
        GlobalIndexToXY(global_index, x, y);
        // std::cout << "Boundary node " << i << " at (x, y) = (" << x << ", " << y << ")\n";

        nodeType_t neighbour;
        for (int q = 0; q < Q; q++)
        {
            const int xn = x + h_cx[q];
            const int yn = y + h_cy[q];

            neighbour = hMom.nodeType[IDX_BLOCK(xn % BLOCK_THREAD_X,
                                                yn % BLOCK_THREAD_Y,
                                                xn / BLOCK_THREAD_X,
                                                yn / BLOCK_THREAD_Y)];

            // if any neighbour is solid the incoming from that node is 0
            if (neighbour == SOLID)
            {
                h_cylinder.incomings[idxBoundPop(i, opp[q])] = 0;
            }
        }

        for (int q = 0; q < Q; q++)
        {
            const int opp_dir = opp[q];
            // outgoing is opposite of the incomings
            if (h_cylinder.incomings[idxBoundPop(i, opp_dir)] == 1)
                h_cylinder.outgoings[idxBoundPop(i, q)] = 1;
            // std::cout << " q=" << q
            //           << " incoming=" << static_cast<int>(h_cylinder.incomings[idxBoundPop(i, q)])
            //           << " outgoing=" << static_cast<int>(h_cylinder.outgoings[idxBoundPop(i, q)]) << "\n";
        }
    }
}

inline void find_incoming_outgoings_bcfluid()
{
    // incoming and outgoings for triangular bcfluids

    constexpr binary_t incomings[4][Q] = {
        {1, 1, 1, 1, 1, 0, 1, 1, 1}, // index 0 - bit 1 is SOLID
        {1, 1, 1, 1, 1, 1, 0, 1, 1}, // index 1 - bit 2 is SOLID
        {1, 1, 1, 1, 1, 1, 1, 1, 0}, // index 2 - bit 4 is SOLID
        {1, 1, 1, 1, 1, 1, 1, 0, 1}  // index 3 - bit 8 is SOLID
    };
    binary_t outgoings[4][Q];
    for (int i = 0; i < 4; i++)
    {
        for (int q = 0; q < Q; q++)
        {
            const int opp_dir = opp[q];
            // outgoing is opposite of the incomings
            if (incomings[i][opp_dir] == 1)
                outgoings[i][q] = 1;
            // std::cout << " i=" << i << " q=" << q
            //           << " incoming=" << static_cast<int>(incomings[i][q])
            //           << " outgoing=" << static_cast<int>(outgoings[i][q]) << "\n";
        }
    }

    // copying this to host constant memory
    memcpy(h_incomings_bcfluid, incomings, sizeof(incomings));
    memcpy(h_outgoings_bcfluid, outgoings, sizeof(outgoings));

    // copying this to device constant memory
    cudaMemcpyToSymbol(d_incomings_bcfluid, incomings, sizeof(incomings));
    cudaMemcpyToSymbol(d_outgoings_bcfluid, outgoings, sizeof(outgoings));
}

inline void write_geometry_files(nodeVar hMom)
{
    std::ofstream xc_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "xc.dat"), std::ios::trunc);
    std::ofstream yc_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "yc.dat"), std::ios::trunc);
    std::ofstream center_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "center.dat"), std::ios::trunc);
    std::ofstream radius_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "radius.dat"), std::ios::trunc);
    std::ofstream solid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "solid.dat"), std::ios::trunc);
    std::ofstream bound_in_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bound.dat"), std::ios::trunc);
    std::ofstream fluid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "fluid.dat"), std::ios::trunc);
    std::ofstream bcfluid_file(construct_path(PATH_FILES, ID_SIM, "grid_layout", "bcfluid.dat"), std::ios::trunc);

    for (int y = 0; y < NY; y++)
    {
        xc_file << XC << " " << y << std::endl;
    }
    for (int x = 0; x < NX; x++)
    {
        yc_file << x << " " << YC << std::endl;
    }

    center_file << XC << " " << YC << " " << std::endl;
    radius_file << 0.5 * D << std::endl;

    for (int y = 0; y < NY; y++)
    {
        for (int x = 0; x < NX; x++)
        {
            unsigned int idx = IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y);

            if (hMom.nodeType[idx] == SOLID)
            {
                solid_file << x << " " << y << " " << static_cast<int>(hMom.nodeType[idx]) << std::endl;
            }
            else if (hMom.nodeType[idx] >= INNER_NODE && hMom.nodeType[idx] < (INNER_NODE + 16))
            {
                bound_in_file << x << " " << y << " " << static_cast<int>(hMom.nodeType[idx]) - INNER_NODE << std::endl;
            }
            else if (hMom.nodeType[idx] >= (BCFLUID_NODE + 0) && hMom.nodeType[idx] <= (BCFLUID_NODE + 3))
            {
                bcfluid_file << x << " " << y << " " << static_cast<int>(hMom.nodeType[idx]) - BCFLUID_NODE << std::endl;
            }
            else if (hMom.nodeType[idx] == BULK)
            {
                fluid_file << x << " " << y << " " << static_cast<int>(hMom.nodeType[idx]) << std::endl;
            }
        }
    }

    // optional: close files (done automatically on destruction)
    solid_file.close();
    bound_in_file.close();
    fluid_file.close();
    // bcfluid_in_file.close();
    // bcfluid_out_file.close();
}

#endif // INITIALIZE_LBM_INLINE_H
