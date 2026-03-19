#pragma once

#include "nodeClass.cuh"

__host__ __device__ inline nodeType_t boundary_definitions(const unsigned int x, const unsigned int y, const unsigned int z)
{
    // Determine boundary flags based on position
    bool isW = (x == 0);
    bool isE = (x == NX - 1);
    bool isS = (y == 0);
    bool isN = (y == NY - 1);
    bool isB = (z == 0);
    bool isF = (z == NZ - 1);

// Apply Periodicity: if a direction is periodic, it's treated as BULK (no boundary)
#if X_PERIODIC
    isW = false;
    isE = false;
#endif
#if Y_PERIODIC
    isS = false;
    isN = false;
#endif
#if Z_PERIODIC
    isB = false;
    isF = false;
#endif

    // --- CORNERS
    if (isN && isW && isF)
        return NORTH_WEST_FRONT;
    if (isN && isW && isB)
        return NORTH_WEST_BACK;
    if (isN && isE && isF)
        return NORTH_EAST_FRONT;
    if (isN && isE && isB)
        return NORTH_EAST_BACK;
    if (isS && isW && isF)
        return SOUTH_WEST_FRONT;
    if (isS && isW && isB)
        return SOUTH_WEST_BACK;
    if (isS && isE && isF)
        return SOUTH_EAST_FRONT;
    if (isS && isE && isB)
        return SOUTH_EAST_BACK;

    // --- EDGES
    if (isN && isW)
        return NORTH_WEST;
    if (isN && isE)
        return NORTH_EAST;
    if (isN && isF)
        return NORTH_FRONT;
    if (isN && isB)
        return NORTH_BACK;

    if (isS && isW)
        return SOUTH_WEST;
    if (isS && isE)
        return SOUTH_EAST;
    if (isS && isF)
        return SOUTH_FRONT;
    if (isS && isB)
        return SOUTH_BACK;

    if (isW && isF)
        return WEST_FRONT;
    if (isW && isB)
        return WEST_BACK;
    if (isE && isF)
        return EAST_FRONT;
    if (isE && isB)
        return EAST_BACK;

    // --- FACES
    if (isN)
        return NORTH;
    if (isS)
        return SOUTH;
    if (isW)
        return WEST;
    if (isE)
        return EAST;
    if (isF)
        return FRONT;
    if (isB)
        return BACK;

    // Default
    return BULK;
}

__device__ inline void current_update_neumaan_density_velocity(const real *s_pop,
                                                               real &rho, real &ux, real &uy, real &uz)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;
    const unsigned int xm1 = (tx + BLOCK_THREAD_X - 1) % BLOCK_THREAD_X;

    real rho_i = 0.0;
    real jx_i = 0.0;
    real jy_i = 0.0;
    real jz_i = 0.0;

    real rho_b = 0.0;
    real jx_b = 0.0;
    real jy_b = 0.0;
    real jz_b = 0.0;

#pragma unroll
    for (int q = 0; q < Q; q++)
    {
        real fi = s_pop[idxPopBlock(xm1, ty, tz, q)];

        rho_i += fi;
        jx_i += fi * toReal(d_cx[q]);
        jy_i += fi * toReal(d_cy[q]);
        jz_i += fi * toReal(d_cz[q]);

        if constexpr (CONVECTIVE_OUTLET)
        {
            real fb = s_pop[idxPopBlock(tx, ty, tz, q)];

            rho_b += fb;
            jx_b += fb * toReal(d_cx[q]);
            jy_b += fb * toReal(d_cy[q]);
            jz_b += fb * toReal(d_cz[q]);
        }
    }
    const real inv_rhoi = toReal(1) / rho_i;
    jx_i *= inv_rhoi;
    jy_i *= inv_rhoi;
    jz_i *= inv_rhoi;

    if constexpr (CONVECTIVE_OUTLET)
    {
        const real UC = d_UCONV;

        const real inv_rhob = toReal(1) / rho_b;
        jx_b *= inv_rhob;
        jy_b *= inv_rhob;
        jz_b *= inv_rhob;

        rho = (1.0 - UC) * rho_b + UC * rho_i;
        // rho = RHO_0;
        ux = (1.0 - UC) * jx_b + UC * jx_i;
        uy = (1.0 - UC) * jy_b + UC * jy_i;
        uz = (1.0 - UC) * jz_b + UC * jz_i;
    }
    else
    {
        rho = rho_i;
        // rho = RHO_0;
        ux = jx_i;
        uy = jy_i;
        uz = jz_i;
    }
}

__device__ inline void update_neumaan_density_velocity(nodeVar dMom, real &rho, real &ux, real &uy, real &uz)
{
    size_t idx;

    // interior node values
    idx = IDX_BLOCK(threadIdx.x - 1, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
    const real rhoi = RHO_0 + dMom.rho[idx];
    const real uxi = dMom.ux[idx];
    const real uyi = dMom.uy[idx];
    const real uzi = dMom.uz[idx];

    if constexpr (CONVECTIVE_OUTLET)
    {
        // boundary node values
        idx = IDX_BLOCK(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
        const real rhob = RHO_0 + dMom.rho[idx];
        const real uxb = dMom.ux[idx];
        const real uyb = dMom.uy[idx];
        const real uzb = dMom.uz[idx];

        // const real Uc = sqrt(uxi * uxi);
        const real Uc = d_UCONV;
        rho = (1.0 - Uc) * rhob + Uc * rhoi;
        // rho = RHO_0;
        ux = (1.0 - Uc) * uxb + Uc * uxi;
        uy = (1.0 - Uc) * uyb + Uc * uyi;
        uz = (1.0 - Uc) * uzb + Uc * uzi;
    }
    else
    {
        rho = rhoi;
        // rho = RHO_0;
        ux = uxi;
        uy = uyi;
        uz = uzi;
    }
}

__device__ inline void boundary_condition(nodeType_t nodeType, nodeVar dMom, real *pop, real *s_pop,
                                          real &rho, real &ux, real &uy, real &uz,
                                          real &mxx, real &myy, real &mzz,
                                          real &mxy, real &mxz, real &myz)
{
    switch (nodeType)
    {
    case WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18] + pop[20] + pop[22] + pop[24] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
        const real mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;

        rho = -toReal(6) * rho_I / (-toReal(5) + toReal(3) * ux + toReal(3) * ux * ux);
        ux = U_MAX;
        uy = toReal(0);
        uz = toReal(0);

        mxx = ux * ux;
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(2) * mxy_I * rho_I / rho;
        mxz = toReal(2) * mxz_I * rho_I / rho;
        myz = toReal(0);

        return;
    }
    case EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        // const real mxx_I = (pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real myy_I = (pop[3] + pop[4] + pop[7] + pop[11] + pop[12] + pop[13] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real mzz_I = (pop[5] + pop[6] + pop[9] + pop[11] + pop[12] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const real mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;
        const real myz_I = (pop[11] + pop[12] - pop[17] - pop[18] + pop[19] - pop[21] - pop[23] + pop[26]) * inv_rho_I;

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        // rho = toReal(6) * rho_I / (toReal(5) + toReal(3) * ux + toReal(3) * ux * ux);

        // mxx = ux * ux;
        // // mxx = (toReal(9) * mxx_I * rho_I + rho - toReal(3) * ux * rho) / (toReal(6) * rho);
        // myy = toReal(6) * myy_I * rho_I / (toReal(5) * rho);
        // mzz = toReal(6) * mzz_I * rho_I / (toReal(5) * rho);
        // mxy = (toReal(6) * mxy_I * rho_I - uy * rho) / (toReal(3) * rho);
        // mxz = (toReal(6) * mxz_I * rho_I - uz * rho) / (toReal(3) * rho);
        // myz = toReal(6) * myz_I * rho_I / (toReal(5) * rho);

        mxx = ux * ux;
        myy = (toReal(6) * myy_I * rho_I - toReal(6) * mzz_I * rho_I +
               toReal(5) * uy * uy * rho + toReal(5) * uz * uz * rho) /
              (toReal(10) * rho);
        mzz = (-toReal(6) * myy_I * rho_I + toReal(6) * mzz_I * rho_I +
               toReal(5) * uy * uy * rho + toReal(5) * uz * uz * rho) /
              (toReal(10) * rho);
        mxy = (toReal(6) * mxy_I * rho_I - uy * rho) / (toReal(3) * rho);
        mxz = (toReal(6) * mxz_I * rho_I - uz * rho) / (toReal(3) * rho);
        myz = toReal(6) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case NORTH:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
        const real myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

        rho = toReal(6) * rho_I / toReal(5);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(2) * mxy_I * rho_I / rho;
        mxz = toReal(0);
        myz = toReal(2) * myz_I * rho_I / rho;

        return;
    }
    case SOUTH:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const real myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

        rho = toReal(6) * rho_I / toReal(5);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(2) * mxy_I * rho_I / rho;
        mxz = toReal(0);
        myz = toReal(2) * myz_I * rho_I / rho;

        return;
    }
    case BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;
        const real myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

        rho = toReal(6) * rho_I / toReal(5);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(2) * mxz_I * rho_I / rho;
        myz = toReal(2) * myz_I * rho_I / rho;

        return;
    }
    case FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;
        const real myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

        rho = toReal(6) * rho_I / toReal(5);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(2) * mxz_I * rho_I / rho;
        myz = toReal(2) * myz_I * rho_I / rho;

        return;
    }
    case NORTH_WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[24] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = -(pop[14] + pop[24] + pop[25]) * inv_rho_I;
        const real myz_I = (pop[11] - pop[17] - pop[24] + pop[25]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + mxy_I - mxy_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I + rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = toReal(12) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case SOUTH_WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18] + pop[20] + pop[22];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = (pop[8] + pop[20] + pop[22]) * inv_rho_I;
        const real myz_I = (pop[12] - pop[18] + pop[20] - pop[22]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxy_I + mxy_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I - rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = toReal(12) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18] + pop[22] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = -(pop[16] + pop[22] + pop[25]) * inv_rho_I;
        const real myz_I = (pop[11] - pop[18] - pop[22] + pop[25]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + mxz_I - mxz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I + rho) / (toReal(9) * rho);
        myz = toReal(12) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17] + pop[20] + pop[24];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = (pop[10] + pop[20] + pop[24]) * inv_rho_I;
        const real myz_I = (pop[12] - pop[17] + pop[20] - pop[24]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxz_I + mxz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I - rho) / (toReal(9) * rho);
        myz = toReal(12) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case NORTH_EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17] + pop[19] + pop[21];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = (pop[7] + pop[19] + pop[21]) * inv_rho_I;
        const real myz_I = (pop[11] - pop[17] + pop[19] - pop[21]) * inv_rho_I;

        // rho = toReal(36) * rho_I * (toReal(1) - mxy_I + mxy_I * OMEGA) / (toReal(24) + OMEGA);
        // rho = toReal(1.5) * rho_I * (toReal(1) - mxy_I);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I - rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = (toReal(12) * myz_I * rho_I) / (toReal(5) * rho);

        return;
    }
    case SOUTH_EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = -(pop[13] + pop[23] + pop[26]) * inv_rho_I;
        const real myz_I = (pop[12] - pop[18] - pop[23] + pop[26]) * inv_rho_I;

        // rho = toReal(36) * rho_I * (toReal(1) + mxy_I - mxy_I * OMEGA) / (toReal(24) + OMEGA);
        // rho = toReal(1.5) * rho_I * (toReal(1) + mxy_I);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I + rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = (toReal(12) * myz_I * rho_I) / (toReal(5) * rho);

        return;
    }
    case EAST_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18] + pop[19] + pop[23];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = (pop[9] + pop[19] + pop[23]) * inv_rho_I;
        const real myz_I = (pop[11] - pop[18] + pop[19] - pop[23]) * inv_rho_I;

        // rho = toReal(36) * rho_I * (toReal(1) - mxz_I + mxz_I * OMEGA) / (toReal(24) + OMEGA);
        // rho = toReal(1.5) * rho_I * (toReal(1) - mxz_I);
        // rho = toReal(36) * rho_I /toReal(25);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I - rho) / (toReal(9) * rho);
        myz = (toReal(12) * myz_I * rho_I) / (toReal(5) * rho);

        return;
    }
    case EAST_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17] + pop[21] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = -(pop[15] + pop[21] + pop[26]) * inv_rho_I;
        const real myz_I = (pop[12] - pop[17] - pop[21] + pop[26]) * inv_rho_I;

        // rho = toReal(36) * rho_I * (toReal(1) + mxz_I - mxz_I * OMEGA) / (toReal(24) + OMEGA);
        // rho = toReal(1.5) * rho_I * (toReal(1) + mxz_I);
        // rho = toReal(36) * rho_I /toReal(25);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I + rho) / (toReal(9) * rho);
        myz = (toReal(12) * myz_I * rho_I) / (toReal(5) * rho);

        return;
    }
    case NORTH_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16] + pop[19] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = (pop[11] + pop[19] + pop[25]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - myz_I + myz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = (toReal(36) * myz_I * rho_I - rho) / (toReal(9) * rho);

        return;
    }
    case NORTH_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17] + pop[21] + pop[24];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = -(pop[17] + pop[21] + pop[24]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + myz_I - myz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = (toReal(36) * myz_I * rho_I + rho) / (toReal(9) * rho);

        return;
    }
    case SOUTH_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18] + pop[22] + pop[23];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = -(pop[18] + pop[22] + pop[23]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + myz_I - myz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = (toReal(36) * myz_I * rho_I + rho) / (toReal(9) * rho);

        return;
    }
    case SOUTH_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[20] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = (pop[12] + pop[20] + pop[26]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - myz_I + myz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = (toReal(36) * myz_I * rho_I - rho) / (toReal(9) * rho);

        return;
    }
    case NORTH_WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16] + pop[25];

        rho = toReal(216) * rho_I / toReal(125);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case NORTH_WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17] + pop[24];

        rho = toReal(216) * rho_I / toReal(125);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18] + pop[22];

        rho = toReal(216) * rho_I / toReal(125);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[20];

        rho = toReal(216) * rho_I / toReal(125);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case NORTH_EAST_FRONT:
    {
        // const real rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[19];

        // rho = toReal(216) * rho_I / toReal(125);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case NORTH_EAST_BACK:
    {
        // const real rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17] + pop[21];

        // rho = toReal(216) * rho_I / toReal(125);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_EAST_FRONT:
    {
        // const real rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18] + pop[23];

        // rho = toReal(216) * rho_I / toReal(125);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_EAST_BACK:
    {
        // const real rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15] + pop[26];

        // rho = toReal(216) * rho_I / toReal(125);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    }
}

__device__ inline void fluid_boundary_condition(const nodeType_t nodeTag, const cylinderVar &cylinder,
                                                const real *pop, real &rho, real &ux, real &uy, real &uz,
                                                real &mxx, real &myy, real &mzz,
                                                real &mxy, real &mxz, real &myz)
{
    real rhoI = toReal(0.0);
    real uxI = toReal(0.0);
    real uyI = toReal(0.0);
    real uzI = toReal(0.0);
    real mxxI = toReal(0.0);
    real myyI = toReal(0.0);
    real mzzI = toReal(0.0);
    real mxyI = toReal(0.0);
    real mxzI = toReal(0.0);
    real myzI = toReal(0.0);

    uint32_t incoming_mask = d_incomingMask_bcfluid[nodeTag];
    while (incoming_mask)
    {
        const int q = __ffs(incoming_mask) - 1;
        incoming_mask &= incoming_mask - 1;

        const real fq = pop[q];
        rhoI += fq;
        uxI += fq * toReal(d_cx[q]);
        uyI += fq * toReal(d_cy[q]);
        uzI += fq * toReal(d_cz[q]);
        mxxI += fq * d_Hxx[q];
        myyI += fq * d_Hyy[q];
        mzzI += fq * d_Hzz[q];
        mxyI += fq * d_Hxy[q];
        mxzI += fq * d_Hxz[q];
        myzI += fq * d_Hyz[q];
    }
    const real inv_rho = toReal(1.0) / rhoI;
    uxI *= inv_rho;
    uyI *= inv_rho;
    uzI *= inv_rho;
    mxxI *= inv_rho;
    myyI *= inv_rho;
    mzzI *= inv_rho;
    mxyI *= inv_rho;
    mxzI *= inv_rho;
    myzI *= inv_rho;

    switch (nodeTag)
    {
    case BCF_1:
    {
        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real a = toReal(-9996) - toReal(206) * OMEGA;
            const real b = toReal(-6) * rhoI * (toReal(12) * mxxI * (toReal(-17) + toReal(20) * OMEGA) - toReal(17) * (toReal(102) + toReal(36) * mxyI + toReal(12) * myyI - toReal(5) * uxI - toReal(5) * uyI) + OMEGA * (toReal(720) * mxyI + toReal(240) * myyI + toReal(53) * (uxI + uyI)));
            const real c = toReal(3) * OMEGA * (rhoI * rhoI) * (toReal(45) * (mxxI * mxxI) + toReal(405) * (mxyI * mxyI) + toReal(45) * (myyI * myyI) + toReal(345) * myyI * uxI + toReal(589) * (uxI * uxI) + toReal(345) * myyI * uyI + toReal(1467) * uxI * uyI + toReal(589) * (uyI * uyI) + toReal(45) * mxyI * (toReal(6) * myyI + toReal(23) * (uxI + uyI)) + toReal(15) * mxxI * (toReal(18) * mxyI + toReal(6) * myyI + toReal(23) * (uxI + uyI)));

            const real disc = max(b * b - toReal(4) * a * c, toReal(0.0));
            const real rho1 = (-b + sqrt(disc)) / (toReal(2) * a);
            const real rho2 = (-b - sqrt(disc)) / (toReal(2) * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
            const real a = toReal(-9998);
            const real b = toReal(6) * rhoI * (toReal(1734) + toReal(66) * mxxI + toReal(198) * mxyI + toReal(66) * myyI + toReal(253) * uxI + toReal(253) * uyI);
            const real c = toReal(3) * (rhoI * rhoI) * (toReal(45) * (mxxI * mxxI) + toReal(405) * (mxyI * mxyI) + toReal(45) * (myyI * myyI) + toReal(345) * myyI * uxI + toReal(589) * (uxI * uxI) + toReal(345) * myyI * uyI + toReal(1467) * uxI * uyI + toReal(589) * (uyI * uyI) + toReal(45) * mxyI * (toReal(6) * myyI + toReal(23) * (uxI + uyI)) + toReal(15) * mxxI * (toReal(18) * mxyI + toReal(6) * myyI + toReal(23) * (uxI + uyI)));

            const real disc = max(b * b - toReal(4) * a * c, toReal(0.0));
            const real rho1 = (-b + sqrt(disc)) / (toReal(2) * a);
            const real rho2 = (-b - sqrt(disc)) / (toReal(2) * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        ux = (toReal(3) * mxxI * rhoI + toReal(9) * mxyI * rhoI + toReal(3) * myyI * rhoI + toReal(20) * rhoI * uxI + toReal(3) * rhoI * uyI + rho) / (toReal(17.) * rho);
        uy = (toReal(3) * mxxI * rhoI + toReal(9) * mxyI * rhoI + toReal(3) * myyI * rhoI + toReal(3) * rhoI * uxI + toReal(20) * rhoI * uyI + rho) / (toReal(17.) * rho);
        uz = (toReal(3) * rhoI * (mxzI + myzI + toReal(10) * uzI)) / (toReal(29.) * rho);
        mxx = (toReal(57) * mxxI * rhoI + toReal(18) * mxyI * rhoI + toReal(6) * myyI * rhoI + toReal(6) * rhoI * uxI + toReal(6) * rhoI * uyI + toReal(2) * rho) / (toReal(51.) * rho);
        myy = (toReal(6) * mxxI * rhoI + toReal(18) * mxyI * rhoI + toReal(57) * myyI * rhoI + toReal(6) * rhoI * uxI + toReal(6) * rhoI * uyI + toReal(2) * rho) / (toReal(51.) * rho);
        mzz = (toReal(36) * mzzI * rhoI) / (toReal(35.) * rho);
        mxy = (toReal(3) * mxxI * rhoI + toReal(26) * mxyI * rhoI + toReal(3) * myyI * rhoI + toReal(3) * rhoI * uxI + toReal(3) * rhoI * uyI + rho) / (toReal(17.) * rho);
        mxz = (rhoI * (toReal(32) * mxzI + toReal(3) * myzI + uzI)) / (toReal(29.) * rho);
        myz = (rhoI * (toReal(3) * mxzI + toReal(32) * myzI + uzI)) / (toReal(29.) * rho);

        break;
    }
    case BCF_2:
    {
        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real a = toReal(9996) + toReal(206) * OMEGA;
            const real b = toReal(6) * rhoI * (mxyI * (toReal(612) - toReal(720) * OMEGA) + toReal(240) * myyI * OMEGA + toReal(12) * mxxI * (toReal(-17) + toReal(20) * OMEGA) - toReal(17) * (toReal(102) + toReal(12) * myyI + toReal(5) * uxI - toReal(5) * uyI) + toReal(53) * OMEGA * (-uxI + uyI));
            const real c = toReal(-3) * OMEGA * (rhoI * rhoI) * (toReal(45) * (mxxI * mxxI) + toReal(405) * (mxyI * mxyI) + toReal(45) * (myyI * myyI) - toReal(345) * myyI * uxI + toReal(589) * (uxI * uxI) + toReal(345) * myyI * uyI - toReal(1467) * uxI * uyI + toReal(589) * (uyI * uyI) - toReal(45) * mxyI * (toReal(6) * myyI - toReal(23) * uxI + toReal(23) * uyI) + toReal(15) * mxxI * (toReal(-18) * mxyI + toReal(6) * myyI - toReal(23) * uxI + toReal(23) * uyI));

            const real disc = max(b * b - toReal(4) * a * c, toReal(0.0));
            const real rho1 = (-b + sqrt(disc)) / (toReal(2) * a);
            const real rho2 = (-b - sqrt(disc)) / (toReal(2) * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
            const real a = toReal(-9998);
            const real b = toReal(6) * rhoI * (toReal(1734) + toReal(66) * mxxI - toReal(198) * mxyI + toReal(66) * myyI - toReal(253) * uxI + toReal(253) * uyI);
            const real c = toReal(3) * (rhoI * rhoI) * (toReal(45) * (mxxI * mxxI) + toReal(405) * (mxyI * mxyI) + toReal(45) * (myyI * myyI) - toReal(345) * myyI * uxI + toReal(589) * (uxI * uxI) + toReal(345) * myyI * uyI - toReal(1467) * uxI * uyI + toReal(589) * (uyI * uyI) - toReal(45) * mxyI * (toReal(6) * myyI - toReal(23) * uxI + toReal(23) * uyI) + toReal(15) * mxxI * (toReal(-18) * mxyI + toReal(6) * myyI - toReal(23) * uxI + toReal(23) * uyI));

            const real disc = max(b * b - toReal(4) * a * c, toReal(0.0));
            const real rho1 = (-b + sqrt(disc)) / (toReal(2) * a);
            const real rho2 = (-b - sqrt(disc)) / (toReal(2) * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        ux = -(toReal(3) * mxxI * rhoI - toReal(9) * mxyI * rhoI + toReal(3) * myyI * rhoI - toReal(20) * rhoI * uxI + toReal(3) * rhoI * uyI + rho) / (toReal(17.) * rho);
        uy = (toReal(3) * mxxI * rhoI - toReal(9) * mxyI * rhoI + toReal(3) * myyI * rhoI - toReal(3) * rhoI * uxI + toReal(20) * rhoI * uyI + rho) / (toReal(17.) * rho);
        uz = (toReal(3) * rhoI * (-mxzI + myzI + toReal(10) * uzI)) / (toReal(29.) * rho);
        mxx = (toReal(57) * mxxI * rhoI - toReal(18) * mxyI * rhoI + toReal(6) * myyI * rhoI - toReal(6) * rhoI * uxI + toReal(6) * rhoI * uyI + toReal(2) * rho) / (toReal(51.) * rho);
        myy = (toReal(6) * mxxI * rhoI - toReal(18) * mxyI * rhoI + toReal(57) * myyI * rhoI - toReal(6) * rhoI * uxI + toReal(6) * rhoI * uyI + toReal(2) * rho) / (toReal(51.) * rho);
        mzz = (toReal(36) * mzzI * rhoI) / (toReal(35.) * rho);
        mxy = -(toReal(3) * mxxI * rhoI - toReal(26) * mxyI * rhoI + toReal(3) * myyI * rhoI - toReal(3) * rhoI * uxI + toReal(3) * rhoI * uyI + rho) / (toReal(17.) * rho);
        mxz = (rhoI * (toReal(32) * mxzI - toReal(3) * myzI - uzI)) / (toReal(29.) * rho);
        myz = (rhoI * (toReal(-3) * mxzI + toReal(32) * myzI + uzI)) / (toReal(29.) * rho);

        break;
    }
    case BCF_3:
    {
        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real a = toReal(9996) + toReal(206) * OMEGA;
            const real b = toReal(6) * rhoI * (mxyI * (toReal(612) - toReal(720) * OMEGA) + toReal(240) * myyI * OMEGA + toReal(12) * mxxI * (toReal(-17) + toReal(20) * OMEGA) + toReal(53) * OMEGA * (uxI - uyI) - toReal(17) * (toReal(102) + toReal(12) * myyI - toReal(5) * uxI + toReal(5) * uyI));
            const real c = toReal(-3) * OMEGA * (rhoI * rhoI) * (toReal(45) * (mxxI * mxxI) + toReal(405) * (mxyI * mxyI) + toReal(45) * (myyI * myyI) + toReal(345) * myyI * uxI + toReal(589) * (uxI * uxI) - toReal(45) * mxyI * (toReal(6) * myyI + toReal(23) * uxI - toReal(23) * uyI) + toReal(15) * mxxI * (toReal(-18) * mxyI + toReal(6) * myyI + toReal(23) * uxI - toReal(23) * uyI) - toReal(345) * myyI * uyI - toReal(1467) * uxI * uyI + toReal(589) * (uyI * uyI));

            const real disc = max(b * b - toReal(4) * a * c, toReal(0.0));
            const real rho1 = (-b + sqrt(disc)) / (toReal(2) * a);
            const real rho2 = (-b - sqrt(disc)) / (toReal(2) * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
            const real a = toReal(-9998);
            const real b = toReal(6) * rhoI * (toReal(1734) + toReal(66) * mxxI - toReal(198) * mxyI + toReal(66) * myyI + toReal(253) * uxI - toReal(253) * uyI);
            const real c = toReal(3) * (rhoI * rhoI) * (toReal(45) * (mxxI * mxxI) + toReal(405) * (mxyI * mxyI) + toReal(45) * (myyI * myyI) + toReal(345) * myyI * uxI + toReal(589) * (uxI * uxI) - toReal(45) * mxyI * (toReal(6) * myyI + toReal(23) * uxI - toReal(23) * uyI) + toReal(15) * mxxI * (toReal(-18) * mxyI + toReal(6) * myyI + toReal(23) * uxI - toReal(23) * uyI) - toReal(345) * myyI * uyI - toReal(1467) * uxI * uyI + toReal(589) * (uyI * uyI));

            const real disc = max(b * b - toReal(4) * a * c, toReal(0.0));
            const real rho1 = (-b + sqrt(disc)) / (toReal(2) * a);
            const real rho2 = (-b - sqrt(disc)) / (toReal(2) * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        ux = (toReal(3) * mxxI * rhoI - toReal(9) * mxyI * rhoI + toReal(3) * myyI * rhoI + toReal(20) * rhoI * uxI - toReal(3) * rhoI * uyI + rho) / (toReal(17.) * rho);
        uy = -(toReal(3) * mxxI * rhoI - toReal(9) * mxyI * rhoI + toReal(3) * myyI * rhoI + toReal(3) * rhoI * uxI - toReal(20) * rhoI * uyI + rho) / (toReal(17.) * rho);
        uz = (toReal(3) * rhoI * (mxzI - myzI + toReal(10) * uzI)) / (toReal(29.) * rho);
        mxx = (toReal(57) * mxxI * rhoI - toReal(18) * mxyI * rhoI + toReal(6) * myyI * rhoI + toReal(6) * rhoI * uxI - toReal(6) * rhoI * uyI + toReal(2) * rho) / (toReal(51.) * rho);
        myy = (toReal(6) * mxxI * rhoI - toReal(18) * mxyI * rhoI + toReal(57) * myyI * rhoI + toReal(6) * rhoI * uxI - toReal(6) * rhoI * uyI + toReal(2) * rho) / (toReal(51.) * rho);
        mzz = (toReal(36) * mzzI * rhoI) / (toReal(35.) * rho);
        mxy = -(toReal(3) * mxxI * rhoI - toReal(26) * mxyI * rhoI + toReal(3) * myyI * rhoI + toReal(3) * rhoI * uxI - toReal(3) * rhoI * uyI + rho) / (toReal(17.) * rho);
        mxz = (rhoI * (toReal(32) * mxzI - toReal(3) * myzI + uzI)) / (toReal(29.) * rho);
        myz = -(rhoI * (toReal(3) * mxzI - toReal(32) * myzI + uzI)) / (toReal(29.) * rho);

        break;
    }
    case BCF_4:
    {
        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real a = toReal(-9996) - toReal(206) * OMEGA;
            const real b = toReal(-6) * rhoI * (toReal(12) * mxxI * (toReal(-17) + toReal(20) * OMEGA) - toReal(17) * (toReal(102) + toReal(36) * mxyI + toReal(12) * myyI + toReal(5) * uxI + toReal(5) * uyI) + OMEGA * (toReal(720) * mxyI + toReal(240) * myyI - toReal(53) * (uxI + uyI)));
            const real c = toReal(3) * OMEGA * (rhoI * rhoI) * (toReal(45) * (mxxI * mxxI) + toReal(405) * (mxyI * mxyI) + toReal(45) * (myyI * myyI) - toReal(345) * myyI * uxI + toReal(589) * (uxI * uxI) - toReal(345) * myyI * uyI + toReal(1467) * uxI * uyI + toReal(589) * (uyI * uyI) + toReal(45) * mxyI * (toReal(6) * myyI - toReal(23) * (uxI + uyI)) + toReal(15) * mxxI * (toReal(18) * mxyI + toReal(6) * myyI - toReal(23) * (uxI + uyI)));

            const real disc = max(b * b - toReal(4) * a * c, toReal(0.0));
            const real rho1 = (-b + sqrt(disc)) / (toReal(2) * a);
            const real rho2 = (-b - sqrt(disc)) / (toReal(2) * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
            const real a = toReal(-9998);
            const real b = toReal(6) * rhoI * (toReal(1734) + toReal(66) * mxxI + toReal(198) * mxyI + toReal(66) * myyI - toReal(253) * uxI - toReal(253) * uyI);
            const real c = toReal(3) * (rhoI * rhoI) * (toReal(45) * (mxxI * mxxI) + toReal(405) * (mxyI * mxyI) + toReal(45) * (myyI * myyI) - toReal(345) * myyI * uxI + toReal(589) * (uxI * uxI) - toReal(345) * myyI * uyI + toReal(1467) * uxI * uyI + toReal(589) * (uyI * uyI) + toReal(45) * mxyI * (toReal(6) * myyI - toReal(23) * (uxI + uyI)) + toReal(15) * mxxI * (toReal(18) * mxyI + toReal(6) * myyI - toReal(23) * (uxI + uyI)));

            const real disc = max(b * b - toReal(4) * a * c, toReal(0.0));
            const real rho1 = (-b + sqrt(disc)) / (toReal(2) * a);
            const real rho2 = (-b - sqrt(disc)) / (toReal(2) * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        ux = -(toReal(3) * mxxI * rhoI + toReal(9) * mxyI * rhoI + toReal(3) * myyI * rhoI - toReal(20) * rhoI * uxI - toReal(3) * rhoI * uyI + rho) / (toReal(17.) * rho);
        uy = -(toReal(3) * mxxI * rhoI + toReal(9) * mxyI * rhoI + toReal(3) * myyI * rhoI - toReal(3) * rhoI * uxI - toReal(20) * rhoI * uyI + rho) / (toReal(17.) * rho);
        uz = (toReal(-3) * rhoI * (mxzI + myzI - toReal(10) * uzI)) / (toReal(29.) * rho);
        mxx = (toReal(57) * mxxI * rhoI + toReal(18) * mxyI * rhoI + toReal(6) * myyI * rhoI - toReal(6) * rhoI * uxI - toReal(6) * rhoI * uyI + toReal(2) * rho) / (toReal(51.) * rho);
        myy = (toReal(6) * mxxI * rhoI + toReal(18) * mxyI * rhoI + toReal(57) * myyI * rhoI - toReal(6) * rhoI * uxI - toReal(6) * rhoI * uyI + toReal(2) * rho) / (toReal(51.) * rho);
        mzz = (toReal(36) * mzzI * rhoI) / (toReal(35.) * rho);
        mxy = (toReal(3) * mxxI * rhoI + toReal(26) * mxyI * rhoI + toReal(3) * myyI * rhoI - toReal(3) * rhoI * uxI - toReal(3) * rhoI * uyI + rho) / (toReal(17.) * rho);
        mxz = (rhoI * (toReal(32) * mxzI + toReal(3) * myzI - uzI)) / (toReal(29.) * rho);
        myz = (rhoI * (toReal(3) * mxzI + toReal(32) * myzI - uzI)) / (toReal(29.) * rho);

        break;
    }
    }
}

__device__ inline void bcsolid_boundary_condition(const nodeType_t nodeTag, const cylinderVar &cylinder,
                                                  const real *pop, real &rho, real &ux, real &uy, real &uz,
                                                  real &mxx, real &myy, real &mzz,
                                                  real &mxy, real &mxz, real &myz)
{
    switch (nodeTag)
    {
    case BCS_1:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18] + pop[19] + pop[23] + pop[25];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxzI = (pop[9] - pop[16] + pop[19] + pop[23] - pop[25]) * inv_rhoI;
        const real myzI = (pop[11] - pop[18] + pop[19] - pop[23] + pop[25]) * inv_rhoI;

        rho = (toReal(108) * (toReal(10) + mxzI * (toReal(-1) + OMEGA) + myzI * (toReal(-1) + OMEGA)) * rhoI) / (toReal(874) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(198) * mxzI * rhoI + toReal(18) * myzI * rhoI - rho) / (toReal(90.) * rho);
        myz = (toReal(18) * mxzI * rhoI + toReal(198) * myzI * rhoI - rho) / (toReal(90.) * rho);
        break;
    }
    case BCS_2:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[14] + pop[16] + pop[18] + pop[19] + pop[22] + pop[25];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxzI = (pop[9] - pop[16] + pop[19] - pop[22] - pop[25]) * inv_rhoI;
        const real myzI = (pop[11] - pop[18] + pop[19] - pop[22] + pop[25]) * inv_rhoI;

        rho = (toReal(108) * (toReal(10) + mxzI + myzI * (toReal(-1) + OMEGA) - mxzI * OMEGA) * rhoI) / (toReal(874) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(198) * mxzI * rhoI - toReal(18) * myzI * rhoI + rho) / (toReal(90.) * rho);
        myz = -(toReal(18) * mxzI * rhoI - toReal(198) * myzI * rhoI + rho) / (toReal(90.) * rho);

        break;
    }
    case BCS_3:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxzI = (pop[9] - pop[16] + pop[19] - pop[22] + pop[23]) * inv_rhoI;
        const real myzI = (pop[11] - pop[18] + pop[19] - pop[22] - pop[23]) * inv_rhoI;

        rho = (toReal(108) * (toReal(10) + myzI + mxzI * (toReal(-1) + OMEGA) - myzI * OMEGA) * rhoI) / (toReal(874) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = -(toReal(-198) * mxzI * rhoI + toReal(18) * myzI * rhoI + rho) / (toReal(90.) * rho);
        myz = (toReal(-18) * mxzI * rhoI + toReal(198) * myzI * rhoI + rho) / (toReal(90.) * rho);

        break;
    }
    case BCS_4:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18] + pop[22] + pop[23] + pop[25];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxzI = (pop[9] - pop[16] - pop[22] + pop[23] - pop[25]) * inv_rhoI;
        const real myzI = (pop[11] - pop[18] - pop[22] - pop[23] + pop[25]) * inv_rhoI;

        rho = (toReal(-108) * (toReal(-10) + mxzI * (toReal(-1) + OMEGA) + myzI * (toReal(-1) + OMEGA)) * rhoI) / (toReal(874) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(198) * mxzI * rhoI + toReal(18) * myzI * rhoI + rho) / (toReal(90.) * rho);
        myz = (toReal(18) * mxzI * rhoI + toReal(198) * myzI * rhoI + rho) / (toReal(90.) * rho);

        break;
    }
    case BCS_5:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[21] + pop[24] + pop[26];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxzI = (pop[10] - pop[15] - pop[21] + pop[24] - pop[26]) * inv_rhoI;
        const real myzI = (pop[12] - pop[17] - pop[21] - pop[24] + pop[26]) * inv_rhoI;

        rho = (toReal(-108) * (toReal(-10) + mxzI * (toReal(-1) + OMEGA) + myzI * (toReal(-1) + OMEGA)) * rhoI) / (toReal(874) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(198) * mxzI * rhoI + toReal(18) * myzI * rhoI + rho) / (toReal(90.) * rho);
        myz = (toReal(18) * mxzI * rhoI + toReal(198) * myzI * rhoI + rho) / (toReal(90.) * rho);

        break;
    }
    case BCS_6:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[14] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxzI = (pop[10] - pop[15] + pop[20] - pop[21] + pop[24]) * inv_rhoI;
        const real myzI = (pop[12] - pop[17] + pop[20] - pop[21] - pop[24]) * inv_rhoI;

        rho = (toReal(108) * (toReal(10) + myzI + mxzI * (toReal(-1) + OMEGA) - myzI * OMEGA) * rhoI) / (toReal(874) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = -(toReal(-198) * mxzI * rhoI + toReal(18) * myzI * rhoI + rho) / (toReal(90.) * rho);
        myz = (toReal(-18) * mxzI * rhoI + toReal(198) * myzI * rhoI + rho) / (toReal(90.) * rho);

        break;
    }
    case BCS_7:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[17] + pop[20] + pop[21] + pop[26];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxzI = (pop[10] - pop[15] + pop[20] - pop[21] - pop[26]) * inv_rhoI;
        const real myzI = (pop[12] - pop[17] + pop[20] - pop[21] + pop[26]) * inv_rhoI;

        rho = (toReal(108) * (toReal(10) + mxzI + myzI * (toReal(-1) + OMEGA) - mxzI * OMEGA) * rhoI) / (toReal(874) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(198) * mxzI * rhoI - toReal(18) * myzI * rhoI + rho) / (toReal(90.) * rho);
        myz = -(toReal(18) * mxzI * rhoI - toReal(198) * myzI * rhoI + rho) / (toReal(90.) * rho);

        break;
    }
    case BCS_8:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[20] + pop[24] + pop[26];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxzI = (pop[10] - pop[15] + pop[20] + pop[24] - pop[26]) * inv_rhoI;
        const real myzI = (pop[12] - pop[17] + pop[20] - pop[24] + pop[26]) * inv_rhoI;

        rho = (toReal(108) * (toReal(10) + mxzI * (toReal(-1) + OMEGA) + myzI * (toReal(-1) + OMEGA)) * rhoI) / (toReal(874) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(198) * mxzI * rhoI + toReal(18) * myzI * rhoI - rho) / (toReal(90.) * rho);
        myz = (toReal(18) * mxzI * rhoI + toReal(198) * myzI * rhoI - rho) / (toReal(90.) * rho);

        break;
    }
    }
}
