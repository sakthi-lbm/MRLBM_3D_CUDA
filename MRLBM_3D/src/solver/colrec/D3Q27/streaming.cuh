#ifndef STREAMING_CUH
#define STREAMING_CUH

__device__ __forceinline__ void streaming(const moments &smem, real *pop)
{
    const unsigned int tx = threadIdx.x + HALO;
    const unsigned int ty = threadIdx.y + HALO;
    const unsigned int tz = threadIdx.z + HALO;

    const real c1 = as2;
    const real c2 = toReal(0.5) * as2 * as2;

#pragma unroll
    for (int q = 0; q < Q; q++)
    {
        const real cx = toReal(d_cx[q]);
        const real cy = toReal(d_cy[q]);
        const real cz = toReal(d_cz[q]);

        int xs = tx - cx;
        int ys = ty - cy;
        int zs = tz - cz;

        const real rho = smem.rho[zs][ys][xs];
        const real ux = smem.ux[zs][ys][xs];
        const real uy = smem.uy[zs][ys][xs];
        const real uz = smem.uz[zs][ys][xs];

        const real mxx = smem.mxx[zs][ys][xs];
        const real myy = smem.myy[zs][ys][xs];
        const real mzz = smem.mzz[zs][ys][xs];
        const real mxy = smem.mxy[zs][ys][xs];
        const real mxz = smem.mxz[zs][ys][xs];
        const real myz = smem.myz[zs][ys][xs];

        const real w = d_w[q];
        const real Hxx = d_Hxx[q];
        const real Hyy = d_Hyy[q];
        const real Hzz = d_Hzz[q];

        const real Hxy = d_Hxy[q];
        const real Hxz = d_Hxz[q];
        const real Hyz = d_Hyz[q];

        const real vel_term = (ux * cx + uy * cy + uz * cz);
        real mom_term = Hxx * mxx + Hyy * myy + Hzz * mzz;

        mom_term += toReal(2.0) * (Hxy * mxy + Hxz * mxz + Hyz * myz);

        pop[q] = w * rho * (toReal(1.0) + c1 * vel_term + c2 * mom_term);
    }
}

// __device__ inline void streaming(const real *s_pop, real *pop)
// {
//     const unsigned int tx = threadIdx.x;
//     const unsigned int ty = threadIdx.y;
//     const unsigned int tz = threadIdx.z;

// #pragma unroll
//     for (int q = 0; q < Q; q++)
//     {
//         int xs = (tx - d_cx[q] + BLOCK_THREAD_X) % BLOCK_THREAD_X;
//         int ys = (ty - d_cy[q] + BLOCK_THREAD_Y) % BLOCK_THREAD_Y;
//         int zs = (tz - d_cz[q] + BLOCK_THREAD_Z) % BLOCK_THREAD_Z;

//         pop[q] = s_pop[idxPopBlock(xs, ys, zs, q)];
//     }
// }

// __device__ inline void save_pop(real *s_pop, const real *pop)
// {
//     // save populations in shared memory
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 0)] = pop[1];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1)] = pop[2];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2)] = pop[3];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3)] = pop[4];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4)] = pop[5];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 5)] = pop[6];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 6)] = pop[7];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7)] = pop[8];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8)] = pop[9];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9)] = pop[10];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = pop[11];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = pop[12];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = pop[13];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = pop[14];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = pop[15];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = pop[16];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = pop[17];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = pop[18];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18)] = pop[19];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19)] = pop[20];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20)] = pop[21];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21)] = pop[22];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22)] = pop[23];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23)] = pop[24];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24)] = pop[25];
//     s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25)] = pop[26];
// }

// __device__ inline void streaming(const real *s_pop, real *pop)
// {
//     const unsigned int xp1 = (threadIdx.x + 1 + BLOCK_THREAD_X) % BLOCK_THREAD_X;
//     const unsigned int xm1 = (threadIdx.x - 1 + BLOCK_THREAD_X) % BLOCK_THREAD_X;

//     const unsigned int yp1 = (threadIdx.y + 1 + BLOCK_THREAD_Y) % BLOCK_THREAD_Y;
//     const unsigned int ym1 = (threadIdx.y - 1 + BLOCK_THREAD_Y) % BLOCK_THREAD_Y;

//     const unsigned int zp1 = (threadIdx.z + 1 + BLOCK_THREAD_Z) % BLOCK_THREAD_Z;
//     const unsigned int zm1 = (threadIdx.z - 1 + BLOCK_THREAD_Z) % BLOCK_THREAD_Z;

//     pop[1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
//     pop[2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
//     pop[3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
//     pop[4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
//     pop[5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
//     pop[6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
//     pop[7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
//     pop[8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
//     pop[9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
//     pop[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
//     pop[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
//     pop[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
//     pop[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
//     pop[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
//     pop[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
//     pop[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
//     pop[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
//     pop[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];
//     pop[19] = s_pop[idxPopBlock(xm1, ym1, zm1, 18)];
//     pop[20] = s_pop[idxPopBlock(xp1, yp1, zp1, 19)];
//     pop[21] = s_pop[idxPopBlock(xm1, ym1, zp1, 20)];
//     pop[22] = s_pop[idxPopBlock(xp1, yp1, zm1, 21)];
//     pop[23] = s_pop[idxPopBlock(xm1, yp1, zm1, 22)];
//     pop[24] = s_pop[idxPopBlock(xp1, ym1, zp1, 23)];
//     pop[25] = s_pop[idxPopBlock(xp1, ym1, zm1, 24)];
//     pop[26] = s_pop[idxPopBlock(xm1, yp1, zp1, 25)];
// }

#endif