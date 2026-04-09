#include "stat_header.cuh"


__global__ void cylinder_force_mass_kernel(const nodeVar dMom,
                                          const size_t *boundaryList,
                                          const size_t *bcfluidList,
                                          const uint32_t *boundaryMask,
                                          const int NB, const int NB_FLUID,
                                          const MaskType masktype,
                                          const real sign)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    real Fx_local = 0.0;
    real Fy_local = 0.0;
    real Fz_local = 0.0;
    real m_local = 0.0;

    if (tid < NB)
    {
        const size_t idx = boundaryList[tid];
        const uint32_t mask = boundaryMask[tid];

        compute_node_force(dMom, idx, mask, Fx_local, Fy_local, Fz_local, m_local);
    }
    else if (tid < NB + NB_FLUID)
    {
        int fid = tid - NB;
        const size_t idx = bcfluidList[fid];

        const nodeType_t nodeType_masked = dMom.nodeType[idx];
        const nodeType_t tag = getIndex(nodeType_masked);

        uint32_t mask;
        if (masktype == INCOMING)
        {
            mask = d_incomingMask_bcfluid[tag];
        }
        else
        {
            mask = d_outgoingMask_bcfluid[tag];
        }
        compute_node_force(dMom, idx, mask, Fx_local, Fy_local, Fz_local, m_local);
    }

    // ================= BLOCK REDUCTION =================
    __shared__ real sFx[BLOCK_NODES];
    __shared__ real sFy[BLOCK_NODES];
    __shared__ real sFz[BLOCK_NODES];
    __shared__ real sm[BLOCK_NODES];

    const unsigned int tx = threadIdx.x;

    sFx[tx] = Fx_local;
    sFy[tx] = Fy_local;
    sFz[tx] = Fz_local;
    sm[tx] = m_local;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tx < stride)
        {
            sFx[tx] += sFx[tx + stride];
            sFy[tx] += sFy[tx + stride];
            sFz[tx] += sFz[tx + stride];
            sm[tx] += sm[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0)
    {
        atomicAdd(&d_TotalFx, sign * sFx[0]);
        atomicAdd(&d_TotalFy, sign * sFy[0]);
        atomicAdd(&d_TotalFz, sign * sFz[0]);
        atomicAdd(&d_Totalm, sign * sm[0]);
    }
}
