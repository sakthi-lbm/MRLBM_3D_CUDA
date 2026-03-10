#ifndef INITIALIZE_LBM_INLINE_H
#define INITIALIZE_LBM_INLINE_H

#include "../utils/geometry_utils.cuh"

inline void check_tau()
{
    if (TAU <= 0.51 || TAU >= 2.5)
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

    checkCudaErrors(cudaMemcpyToSymbol(d_NB, &NB, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_NB_FLUID, &NB_FLUID, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_NB_SOLID, &d_NB_SOLID, sizeof(int)));
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

inline void buildBoundaryList_updateBoundaryNodeType(nodeVar &hMom, cylinderVar &h_cylinder)
{
    int count = 0;
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

inline void find_incomings_outgoings_cylinder(const nodeVar &hMom, cylinderVar &h_cylinder, const int nb)
{
    if (nb <= 0)
        return;

    for (int i = 0; i < nb; i++)
    {
        uint32_t incomingMask = (1u << Q) - 1;
        uint32_t outgoingMask = 0;

        const size_t global_index = h_cylinder.boundaryList[i];
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
        h_cylinder.incomingMask[i] = incomingMask;
        h_cylinder.outgoingMask[i] = outgoingMask;

        // // debuggig
        // if (z == 0)
        // {
        //     std::cout << "Boundary node " << i << " at (x, y,z) = (" << x << ", " << y << ")\n";
        //     for (int q = 0; q < Q; q++)
        //     {
        //         binary_t incomingMaskBit = (incomingMask >> q) & 1u;
        //         binary_t outgoingMaskBit = (outgoingMask >> q) & 1u;

        //         std::cout << " q=" << q
        //                   << " incomingMask=" << static_cast<int>(incomingMaskBit)
        //                   << " outgoingMask=" << static_cast<int>(outgoingMaskBit)
        //                   << "\n";
        //     }
        // }
    }
}

#endif // INITIALIZE_LBM_INLINE_H
