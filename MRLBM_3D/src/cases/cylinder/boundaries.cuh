#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "../../nodeTypeMap.h"
#include "constants.h"
#include "../../globalStructs.h"

__host__ __device__ inline nodeType_t boundary_definitions(const int x, const int y)
{
    // clang-format off
    if (x == 0 && y == 0)
    {
        #if X_PERIODIC && Y_PERIODIC
                return BULK;
        #elif X_PERIODIC
                return SOUTH;
        #elif Y_PERIODIC
                return WEST;
        #else
                return SOUTH_WEST;
        #endif
    }
    else if (x == 0 && y == (NY - 1))
    {
        #if X_PERIODIC && Y_PERIODIC
                return BULK;
        #elif X_PERIODIC
                return NORTH;
        #elif Y_PERIODIC
                return WEST;
        #else
                return NORTH_WEST;
        #endif
    }
    else if (x == (NX - 1) && y == 0)
    {
        #if X_PERIODIC && Y_PERIODIC
                return BULK;
        #elif X_PERIODIC
                return SOUTH;
        #elif Y_PERIODIC
                return EAST;
        #else
                return SOUTH_EAST;
        #endif
    }
    else if (x == (NX - 1) && y == (NY - 1))
    {
        #if X_PERIODIC && Y_PERIODIC
                return BULK;
        #elif X_PERIODIC
                return NORTH;
        #elif Y_PERIODIC
                return EAST;
        #else
                return NORTH_EAST;
        #endif
    }
    else if (x == 0)
    {
        #if X_PERIODIC
                return BULK;
        #else
                return WEST;
        #endif
    }
    else if (x == (NX - 1))
    {
        #if X_PERIODIC
                return BULK;
        #else
                return EAST;
        #endif
    }
    else if (y == 0)
    {
        #if Y_PERIODIC
                return BULK;
        #else
                return SOUTH;
        #endif
    }
    else if (y == (NY - 1))
    {
        #if Y_PERIODIC
                return BULK;
        #else
                return NORTH;
        #endif
    }
    else
    {
        return BULK;
    }

    // clang-format on
}

__device__ inline void boundary_condition(nodeType_t nodeType, nodeVar fMom, real *pop, real &rhoVar, real &ux, real &uy, real &mxx, real &myy, real &mxy)
{
    switch (nodeType)
    {
    case NORTH:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxxI = (pop[1] + pop[3] + pop[5] + pop[6]) * inv_rhoI - cs2;
        const real myyI = (pop[2] + pop[5] + pop[6]) * inv_rhoI - cs2;
        const real mxyI = (pop[5] - pop[6]) * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        const real numerator = 3.0 * (-4.0 + 3.0 * myyI * (OMEGA - 1.0)) * rhoI;
        const real denominator = 3.0 * (-3.0 + uy) + OMEGA * (-1.0 + 3.0 * uy + 6.0 * uy * uy);

        rhoVar = numerator / denominator;

        mxx = 6.0 * rhoI * mxxI / (5.0 * (rhoVar));
        myy = (9.0 * rhoI * myyI + rhoVar - 3.0 * uy * rhoVar) / (6.0 * (rhoVar));
        mxy = (6.0 * rhoI * mxyI - ux * rhoVar) / (3.0 * rhoVar);

        break;
    }
    case SOUTH:
    {
        const real rhoI = pop[0] + pop[1] + pop[3] + pop[4] + pop[7] + pop[8];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxxI = (pop[1] + pop[3] + pop[7] + pop[8]) * inv_rhoI - cs2;
        const real myyI = (pop[4] + pop[7] + pop[8]) * inv_rhoI - cs2;
        const real mxyI = (pop[7] - pop[8]) * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        const real numerator = 3.0 * (4.0 - 3.0 * myyI * (OMEGA - 1.0)) * rhoI;
        const real denominator = 3.0 * (3.0 + uy) + OMEGA * (1.0 + 3.0 * uy - 6.0 * uy * uy);

        rhoVar = numerator / denominator;

        mxx = 6.0 * rhoI * mxxI / (5.0 * (rhoVar));
        myy = (rhoVar + 9.0 * rhoI * myyI + 3.0 * uy * rhoVar) / (6.0 * (rhoVar));
        mxy = (6.0 * rhoI * mxyI + ux * rhoVar) / (3.0 * rhoVar);

        break;
    }
    case WEST:
    {

        const real rhoI = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxxI = (pop[3] + pop[6] + pop[7]) * inv_rhoI - cs2;
        const real myyI = (pop[2] + pop[4] + pop[6] + pop[7]) * inv_rhoI - cs2;
        const real mxyI = (pop[7] - pop[6]) * inv_rhoI;

        ux = U_MAX;
        uy = 0.0;

        rhoVar = ((4.0 + 3.0 * mxxI) * rhoI) / (3.0 - 3.0 * ux);

        mxx = (9.0 * rhoI * mxxI + rhoVar + 3.0 * ux * rhoVar) / (6.0 * (rhoVar));
        myy = (6.0 * rhoI * myyI) / (5.0 * (rhoVar));
        mxy = (6.0 * rhoI * mxyI + uy * rhoVar) / (3.0 * rhoVar);

        break;
    }
    case EAST:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxxI = (pop[1] + pop[5] + pop[8]) * inv_rhoI - cs2;
        const real myyI = (pop[2] + pop[4] + pop[5] + pop[8]) * inv_rhoI - cs2;
        const real mxyI = (pop[5] - pop[8]) * inv_rhoI;

        // size_t idx = IDX_BLOCK(threadIdx.x - 1, threadIdx.y, blockIdx.x, blockIdx.y);
        // rhoVar = RHO_0 + fMom.rho[idx];
        // ux = fMom.ux[idx];
        // uy = fMom.uy[idx];

        const real rhob = RHO_0 + fMom.rho[IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y)];
        const real uxb = fMom.ux[IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y)];
        const real uyb = fMom.uy[IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y)];

        const real rhoi = RHO_0 + fMom.rho[IDX_BLOCK(threadIdx.x - 1, threadIdx.y, blockIdx.x, blockIdx.y)];
        const real uxi = fMom.ux[IDX_BLOCK(threadIdx.x - 1, threadIdx.y, blockIdx.x, blockIdx.y)];
        const real uyi = fMom.uy[IDX_BLOCK(threadIdx.x - 1, threadIdx.y, blockIdx.x, blockIdx.y)];

        rhoVar = (1.0 - U_MAX) * rhob + U_MAX * rhoi;
        ux = (1.0 - U_MAX) * uxb + U_MAX * uxi;
        uy = (1.0 - U_MAX) * uyb + U_MAX * uyi;

        mxx = (9.0 * rhoI * mxxI + rhoVar - 3.0 * ux * rhoVar) / (6.0 * (rhoVar));
        myy = (6.0 * rhoI * myyI) / (5.0 * (rhoVar));
        mxy = (6.0 * mxyI * rhoI - uy * rhoVar) / (3.0 * rhoVar);

        break;
    }
    case SOUTH_WEST:
    {

        const real rhoI = pop[0] + pop[3] + pop[4] + pop[7];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxxI = (pop[3] + pop[7]) * inv_rhoI - cs2;
        const real myyI = (pop[4] + pop[7]) * inv_rhoI - cs2;
        const real mxyI = pop[7] * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        const real numerator = 12.0 * rhoI * (-3.0 - 3.0 * myyI + 3.0 * mxxI * (OMEGA - 1.0) - 7.0 * mxyI * (OMEGA - 1.0) + 3.0 * myyI * OMEGA);
        const real denominator = -2.0 * (8.0 + 7.0 * ux + 7.0 * uy) + OMEGA * (-9.0 + 15.0 * ux * ux - uy + 15.0 * uy * uy - ux * (1.0 + 9.0 * uy));

        rhoVar = numerator / denominator;

        mxx = 2.0 * (9.0 * mxxI * rhoI - 6.0 * mxyI * rhoI + rhoVar + 2.0 * ux * rhoVar - uy * rhoVar) / (9.0 * rhoVar);
        myy = -2.0 * (6.0 * mxyI * rhoI - 9.0 * myyI * rhoI - rhoVar + ux * rhoVar - 2.0 * uy * rhoVar) / (9.0 * rhoVar);
        mxy = -(18.0 * mxxI * rhoI - 132.0 * mxyI * rhoI + 18.0 * myyI * rhoI + 7.0 * rhoVar - 7.0 * ux * rhoVar - 7.0 * uy * rhoVar) / (27.0 * rhoVar);

        break;
    }
    case SOUTH_EAST:
    {
        const real rhoI = pop[0] + pop[1] + pop[4] + pop[8];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxxI = (pop[1] + pop[8]) * inv_rhoI - cs2;
        const real myyI = (pop[4] + pop[8]) * inv_rhoI - cs2;
        const real mxyI = -pop[8] * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        const real numerator = 12.0 * rhoI * (-3.0 - 3.0 * myyI + 3.0 * mxxI * (OMEGA - 1.0) + 7.0 * mxyI * (OMEGA - 1.0) + 3.0 * myyI * OMEGA);
        const real denominator = 2.0 * (-8.0 + 7.0 * ux - 7.0 * uy) + OMEGA * (-9.0 + ux + 15.0 * ux * ux - uy + 9.0 * ux * uy + 15.0 * uy * uy);

        rhoVar = numerator / denominator;

        mxx = -2.0 * (-9.0 * mxxI * rhoI - 6.0 * mxyI * rhoI - rhoVar + 2.0 * ux * rhoVar + uy * rhoVar) / (9.0 * rhoVar);
        myy = 2.0 * (6.0 * mxyI * rhoI + 9.0 * myyI * rhoI + rhoVar + ux * rhoVar + 2.0 * uy * rhoVar) / (9.0 * rhoVar);
        mxy = -(-18.0 * mxxI * rhoI - 132.0 * mxyI * rhoI - 18.0 * myyI * rhoI - 7.0 * rhoVar - 7.0 * ux * rhoVar + 7.0 * uy * rhoVar) / (27.0 * rhoVar);

        break;
    }
    case NORTH_WEST:
    {
        const real rhoI = pop[0] + pop[2] + pop[3] + pop[6];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxxI = (pop[3] + pop[6]) * inv_rhoI - cs2;
        const real myyI = (pop[2] + pop[6]) * inv_rhoI - cs2;
        const real mxyI = -pop[6] * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        const real numerator = 12.0 * rhoI * (-3.0 - 3.0 * myyI + 3.0 * mxxI * (OMEGA - 1.0) + 7.0 * mxyI * (OMEGA - 1.0) + 3.0 * myyI * OMEGA);
        const real denominator = -2.0 * (8.0 + 7.0 * ux - 7.0 * uy) + OMEGA * (-9.0 + 15.0 * ux * ux + uy + 15.0 * uy * uy + ux * (-1.0 + 9.0 * uy));

        rhoVar = numerator / denominator;

        mxx = 2.0 * (9.0 * mxxI * rhoI + 6.0 * mxyI * rhoI + rhoVar + 2.0 * ux * rhoVar + uy * rhoVar) / (9.0 * rhoVar);
        myy = -2.0 * (-6.0 * mxyI * rhoI - 9.0 * myyI * rhoI - rhoVar + ux * rhoVar + 2.0 * uy * rhoVar) / (9.0 * rhoVar);
        mxy = -(-18.0 * mxxI * rhoI - 132.0 * mxyI * rhoI - 18.0 * myyI * rhoI - 7.0 * rhoVar + 7.0 * ux * rhoVar - 7.0 * uy * rhoVar) / (27.0 * rhoVar);
    }
    case NORTH_EAST:
    {
        const real rhoI = pop[0] + pop[1] + pop[2] + pop[5];
        const real inv_rhoI = 1.0 / rhoI;
        const real mxxI = (pop[1] + pop[5]) * inv_rhoI - cs2;
        const real myyI = (pop[2] + pop[5]) * inv_rhoI - cs2;
        const real mxyI = pop[5] * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        const real numerator = 12.0 * rhoI * (-3.0 - 3.0 * myyI + 3.0 * mxxI * (OMEGA - 1.0) - 7.0 * mxyI * (OMEGA - 1.0) + 3.0 * myyI * OMEGA);
        const real denominator = 2.0 * (-8.0 + 7.0 * ux + 7.0 * uy) + OMEGA * (-9.0 + ux + 15.0 * ux * ux + uy - 9.0 * ux * uy + 15.0 * uy * uy);

        rhoVar = numerator / denominator;

        mxx = -2.0 * (-9.0 * mxxI * rhoI + 6.0 * mxyI * rhoI - rhoVar + 2.0 * ux * rhoVar - uy * rhoVar) / (9.0 * rhoVar);
        myy = 2.0 * (-6.0 * mxyI * rhoI + 9.0 * myyI * rhoI + rhoVar + ux * rhoVar - 2.0 * uy * rhoVar) / (9.0 * rhoVar);
        mxy = -(18.0 * mxxI * rhoI - 132.0 * mxyI * rhoI + 18.0 * myyI * rhoI + 7.0 * rhoVar + 7.0 * ux * rhoVar + 7.0 * uy * rhoVar) / (27.0 * rhoVar);

        break;
    }

    default:
        break;
    }
}

__device__ inline void fluid_boundary_condition(const nodeType_t nodeTag,
                                                const real *pop, real &rhoVar, real &ux, real &uy,
                                                real &mxx, real &myy, real &mxy)
{
    if (nodeTag == 0)
    {
        // const int incomings[Q] = {1, 1, 1, 1, 1, 0, 1, 1, 1};
        // const int outgoings[Q] = {1, 1, 1, 1, 1, 1, 1, 0, 1};

        const real omega = OMEGA;

        real rhoI = toReal(0.0);
        real uxI = toReal(0.0);
        real uyI = toReal(0.0);
        real mxxI = toReal(0.0);
        real myyI = toReal(0.0);
        real mxyI = toReal(0.0);

        for (int q = 0; q < Q; q++)
        {
            if (d_incomings_bcfluid[0][q] == 1)
            {
                const real cx = d_cx[q];
                const real cy = d_cy[q];

                const real Hxx = cx * cx - cs2;
                const real Hyy = cy * cy - cs2;
                const real Hxy = cx * cy;

                rhoI += pop[q];
                uxI += pop[q] * cx;
                uyI += pop[q] * cy;
                mxxI += pop[q] * Hxx;
                myyI += pop[q] * Hyy;
                mxyI += pop[q] * Hxy;
            }
        }
        if (rhoI <= 0.0)
            return;
        const real inv_rho = toReal(1.0) / rhoI;

        uxI *= inv_rho;
        uyI *= inv_rho;
        mxxI *= inv_rho;
        myyI *= inv_rho;
        mxyI *= inv_rho;

        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real linear_part = (5202.0 * rhoI + 612.0 * mxxI * rhoI + 1836.0 * mxyI * rhoI +
                                      612.0 * myyI * rhoI - 720.0 * mxxI * omega * rhoI -
                                      2160.0 * mxyI * omega * rhoI - 720.0 * myyI * omega * rhoI -
                                      255.0 * rhoI * uxI - 159.0 * omega * rhoI * uxI -
                                      255.0 * rhoI * uyI - 159.0 * omega * rhoI * uyI);

            const real inner_expr = (-1734.0 - 204.0 * myyI + 240.0 * myyI * omega +
                                     12.0 * mxxI * (-17.0 + 20.0 * omega) +
                                     36.0 * mxyI * (-17.0 + 20.0 * omega) +
                                     85.0 * uxI + 53.0 * omega * uxI +
                                     85.0 * uyI + 53.0 * omega * uyI);

            const real squared_term = 3.0 * inner_expr * inner_expr;

            const real quadratic_terms = (45.0 * mxxI * mxxI + 405.0 * mxyI * mxyI + 45.0 * myyI * myyI +
                                          345.0 * myyI * uxI + 589.0 * uxI * uxI +
                                          345.0 * myyI * uyI + 1467.0 * uxI * uyI +
                                          589.0 * uyI * uyI +
                                          45.0 * mxyI * (6.0 * myyI + 23.0 * (uxI + uyI)) +
                                          15.0 * mxxI * (18.0 * mxyI + 6.0 * myyI + 23.0 * (uxI + uyI)));

            const real omega_factor = 2.0 * omega * (4998.0 + 103.0 * omega);

            const real sqrt_expr = sqrt(3.0 * rhoI * rhoI * (squared_term + omega_factor * quadratic_terms));

            rhoVar = (linear_part + sqrt_expr) / (9996.0 + 206.0 * omega);
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
            const real linear_part = 3.0 * rhoI * (1734.0 + 66.0 * mxxI + 198.0 * mxyI + 66.0 * myyI + 253.0 * uxI + 253.0 * uyI);
            const real inner_rho_sq = (31212.0 + 1602.0 * mxxI * mxxI + 14418.0 * mxyI * mxyI + 2376.0 * myyI +
                                       1602.0 * myyI * myyI + 9108.0 * uxI + 12282.0 * myyI * uxI +
                                       21041.0 * uxI * uxI + 9108.0 * uyI + 12282.0 * myyI * uyI +
                                       52080.0 * uxI * uyI + 21041.0 * uyI * uyI +
                                       18.0 * mxyI * (396.0 + 534.0 * myyI + 2047.0 * uxI + 2047.0 * uyI) +
                                       6.0 * mxxI * (396.0 + 1602.0 * mxyI + 534.0 * myyI + 2047.0 * uxI + 2047.0 * uyI));

            const real sqrt_part = 17.0 * SQRT_3 * std::sqrt(rhoI * rhoI * inner_rho_sq);

            rhoVar = (linear_part + sqrt_part) / 9998.0;
        }

        ux = -(-3.0 * mxxI * rhoI - 9.0 * mxyI * rhoI - 3.0 * myyI * rhoI -
               20.0 * rhoI * uxI - 3.0 * rhoI * uyI - rhoVar) /
             (17.0 * rhoVar);

        uy = -(-3.0 * mxxI * rhoI - 9.0 * mxyI * rhoI - 3.0 * myyI * rhoI -
               3.0 * rhoI * uxI - 20.0 * rhoI * uyI - rhoVar) /
             (17.0 * rhoVar);

        mxx = -(-57.0 * mxxI * rhoI - 18.0 * mxyI * rhoI - 6.0 * myyI * rhoI -
                6.0 * rhoI * uxI - 6.0 * rhoI * uyI - 2.0 * rhoVar) /
              (51.0 * rhoVar);

        myy = (6.0 * mxxI * rhoI + 18.0 * mxyI * rhoI + 57.0 * myyI * rhoI +
               6.0 * rhoI * uxI + 6.0 * rhoI * uyI + 2.0 * rhoVar) /
              (51.0 * rhoVar);

        mxy = (3.0 * mxxI * rhoI + 26.0 * mxyI * rhoI + 3.0 * myyI * rhoI +
               3.0 * rhoI * uxI + 3.0 * rhoI * uyI + rhoVar) /
              (17.0 * rhoVar);
    }
    else if (nodeTag == 1)
    {
        // const int incomings[Q] = {1, 1, 1, 1, 1, 1, 0, 1, 1};
        // const int outgoings[Q] = {1, 1, 1, 1, 1, 1, 1, 1, 0};

        const real omega = OMEGA;

        real rhoI = toReal(0.0);
        real uxI = toReal(0.0);
        real uyI = toReal(0.0);
        real mxxI = toReal(0.0);
        real myyI = toReal(0.0);
        real mxyI = toReal(0.0);
        for (int q = 0; q < Q; q++)
        {
            if (d_incomings_bcfluid[1][q] == 1)
            {
                const real cx = d_cx[q];
                const real cy = d_cy[q];

                const real Hxx = cx * cx - cs2;
                const real Hyy = cy * cy - cs2;
                const real Hxy = cx * cy;

                rhoI += pop[q];
                uxI += pop[q] * cx;
                uyI += pop[q] * cy;
                mxxI += pop[q] * Hxx;
                myyI += pop[q] * Hyy;
                mxyI += pop[q] * Hxy;
            }
        }
        if (rhoI <= 0.0)
            return;
        const real inv_rho = toReal(1.0) / rhoI;

        uxI *= inv_rho;
        uyI *= inv_rho;
        mxxI *= inv_rho;
        myyI *= inv_rho;
        mxyI *= inv_rho;

        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real linear_part = (5202.0 + 612.0 * mxxI - 1836.0 * mxyI + 612.0 * myyI -
                                      720.0 * mxxI * omega + 2160.0 * mxyI * omega - 720.0 * myyI * omega +
                                      255.0 * uxI + 159.0 * omega * uxI - 255.0 * uyI - 159.0 * omega * uyI) *
                                     rhoI;

            const real inner_expr = (1734.0 + 204.0 * myyI + mxxI * (204.0 - 240.0 * omega) -
                                     240.0 * myyI * omega + 36.0 * mxyI * (-17.0 + 20.0 * omega) +
                                     85.0 * uxI + 53.0 * omega * uxI - 85.0 * uyI - 53.0 * omega * uyI);

            const real squared_term = 3.0 * inner_expr * inner_expr;

            const real quadratic_terms = (45.0 * mxxI * mxxI + 405.0 * mxyI * mxyI + 45.0 * myyI * myyI -
                                          345.0 * myyI * uxI + 589.0 * uxI * uxI -
                                          15.0 * mxxI * (18.0 * mxyI - 6.0 * myyI + 23.0 * uxI - 23.0 * uyI) +
                                          345.0 * myyI * uyI - 1467.0 * uxI * uyI + 589.0 * uyI * uyI -
                                          45.0 * mxyI * (6.0 * myyI - 23.0 * uxI + 23.0 * uyI));

            const real omega_factor = 2.0 * omega * (4998.0 + 103.0 * omega);

            const real sqrt_expr = sqrt(3.0 * rhoI * rhoI * (squared_term + omega_factor * quadratic_terms));

            rhoVar = (linear_part + sqrt_expr) / (9996.0 + 206.0 * omega);
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
            const real linear_part = 3.0 * rhoI * (1734.0 + 66.0 * mxxI - 198.0 * mxyI + 66.0 * myyI - 253.0 * uxI + 253.0 * uyI);
            const real inner_rho_sq = (31212.0 + 1602.0 * mxxI * mxxI + 14418.0 * mxyI * mxyI +
                                       2376.0 * myyI + 1602.0 * myyI * myyI - 9108.0 * uxI -
                                       12282.0 * myyI * uxI + 21041.0 * uxI * uxI -
                                       6.0 * mxxI * (-396.0 + 1602.0 * mxyI - 534.0 * myyI + 2047.0 * uxI - 2047.0 * uyI) +
                                       9108.0 * uyI + 12282.0 * myyI * uyI - 52080.0 * uxI * uyI +
                                       21041.0 * uyI * uyI - 18.0 * mxyI * (396.0 + 534.0 * myyI - 2047.0 * uxI + 2047.0 * uyI));

            const real sqrt_part = 17.0 * SQRT_3 * std::sqrt(rhoI * rhoI * inner_rho_sq);

            rhoVar = (linear_part + sqrt_part) / 9998.0;
        }

        ux = -(3.0 * mxxI * rhoI - 9.0 * mxyI * rhoI + 3.0 * myyI * rhoI -
               20.0 * rhoI * uxI + 3.0 * rhoI * uyI + rhoVar) /
             (17.0 * rhoVar);

        uy = -(-3.0 * mxxI * rhoI + 9.0 * mxyI * rhoI - 3.0 * myyI * rhoI +
               3.0 * rhoI * uxI - 20.0 * rhoI * uyI - rhoVar) /
             (17.0 * rhoVar);

        mxx = -(-57.0 * mxxI * rhoI + 18.0 * mxyI * rhoI - 6.0 * myyI * rhoI +
                6.0 * rhoI * uxI - 6.0 * rhoI * uyI - 2.0 * rhoVar) /
              (51.0 * rhoVar);

        myy = (6.0 * mxxI * rhoI - 18.0 * mxyI * rhoI + 57.0 * myyI * rhoI -
               6.0 * rhoI * uxI + 6.0 * rhoI * uyI + 2.0 * rhoVar) /
              (51.0 * rhoVar);

        mxy = -(3.0 * mxxI * rhoI - 26.0 * mxyI * rhoI + 3.0 * myyI * rhoI -
                3.0 * rhoI * uxI + 3.0 * rhoI * uyI + rhoVar) /
              (17.0 * rhoVar);
    }
    else if (nodeTag == 2)
    {

        // const int incomings[Q] = {1, 1, 1, 1, 1, 1, 1, 1, 0};
        // const int outgoings[Q] = {1, 1, 1, 1, 1, 1, 0, 1, 1};

        const real omega = OMEGA;

        real rhoI = toReal(0.0);
        real uxI = toReal(0.0);
        real uyI = toReal(0.0);
        real mxxI = toReal(0.0);
        real myyI = toReal(0.0);
        real mxyI = toReal(0.0);
        for (int q = 0; q < Q; q++)
        {
            if (d_incomings_bcfluid[2][q] == 1)
            {
                const real cx = d_cx[q];
                const real cy = d_cy[q];

                const real Hxx = cx * cx - cs2;
                const real Hyy = cy * cy - cs2;
                const real Hxy = cx * cy;

                rhoI += pop[q];
                uxI += pop[q] * cx;
                uyI += pop[q] * cy;
                mxxI += pop[q] * Hxx;
                myyI += pop[q] * Hyy;
                mxyI += pop[q] * Hxy;
            }
        }
        if (rhoI <= 0.0)
            return;
        const real inv_rho = toReal(1.0) / rhoI;

        uxI *= inv_rho;
        uyI *= inv_rho;
        mxxI *= inv_rho;
        myyI *= inv_rho;
        mxyI *= inv_rho;

        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real linear_part = (5202.0 * rhoI + 612.0 * mxxI * rhoI - 1836.0 * mxyI * rhoI +
                                      612.0 * myyI * rhoI - 720.0 * mxxI * omega * rhoI +
                                      2160.0 * mxyI * omega * rhoI - 720.0 * myyI * omega * rhoI -
                                      255.0 * rhoI * uxI - 159.0 * omega * rhoI * uxI +
                                      255.0 * rhoI * uyI + 159.0 * omega * rhoI * uyI);

            const real inner_expr = (1734.0 + 204.0 * myyI + mxxI * (204.0 - 240.0 * omega) -
                                     240.0 * myyI * omega + 36.0 * mxyI * (-17.0 + 20.0 * omega) -
                                     85.0 * uxI - 53.0 * omega * uxI + 85.0 * uyI + 53.0 * omega * uyI);

            const real squared_term = 3.0 * inner_expr * inner_expr;

            const real quadratic_terms = (45.0 * mxxI * mxxI + 405.0 * mxyI * mxyI + 45.0 * myyI * myyI +
                                          345.0 * myyI * uxI + 589.0 * uxI * uxI -
                                          45.0 * mxyI * (6.0 * myyI + 23.0 * uxI - 23.0 * uyI) -
                                          345.0 * myyI * uyI - 1467.0 * uxI * uyI + 589.0 * uyI * uyI -
                                          15.0 * mxxI * (18.0 * mxyI - 6.0 * myyI - 23.0 * uxI + 23.0 * uyI));

            const real omega_factor = 2.0 * omega * (4998.0 + 103.0 * omega);

            const real sqrt_expr = sqrt(3.0 * rhoI * rhoI * (squared_term + omega_factor * quadratic_terms));

            rhoVar = (linear_part + sqrt_expr) / (9996.0 + 206.0 * omega);
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
            const real linear_part = 3.0 * rhoI * (1734.0 + 66.0 * mxxI - 198.0 * mxyI + 66.0 * myyI + 253.0 * uxI - 253.0 * uyI);
            const real inner_rho_sq = (31212.0 + 1602.0 * mxxI * mxxI + 14418.0 * mxyI * mxyI +
                                       2376.0 * myyI + 1602.0 * myyI * myyI + 9108.0 * uxI +
                                       12282.0 * myyI * uxI + 21041.0 * uxI * uxI -
                                       18.0 * mxyI * (396.0 + 534.0 * myyI + 2047.0 * uxI - 2047.0 * uyI) -
                                       9108.0 * uyI - 12282.0 * myyI * uyI - 52080.0 * uxI * uyI +
                                       21041.0 * uyI * uyI - 6.0 * mxxI * (-396.0 + 1602.0 * mxyI - 534.0 * myyI - 2047.0 * uxI + 2047.0 * uyI));

            const real sqrt_part = 17.0 * SQRT_3 * std::sqrt(rhoI * rhoI * inner_rho_sq);

            rhoVar = (linear_part + sqrt_part) / 9998.0;
        }

        ux = -(-3.0 * mxxI * rhoI + 9.0 * mxyI * rhoI - 3.0 * myyI * rhoI -
               20.0 * rhoI * uxI + 3.0 * rhoI * uyI - rhoVar) /
             (17.0 * rhoVar);

        uy = -(3.0 * mxxI * rhoI - 9.0 * mxyI * rhoI + 3.0 * myyI * rhoI +
               3.0 * rhoI * uxI - 20.0 * rhoI * uyI + rhoVar) /
             (17.0 * rhoVar);

        mxx = -(-57.0 * mxxI * rhoI + 18.0 * mxyI * rhoI - 6.0 * myyI * rhoI -
                6.0 * rhoI * uxI + 6.0 * rhoI * uyI - 2.0 * rhoVar) /
              (51.0 * rhoVar);

        myy = (6.0 * mxxI * rhoI - 18.0 * mxyI * rhoI + 57.0 * myyI * rhoI +
               6.0 * rhoI * uxI - 6.0 * rhoI * uyI + 2.0 * rhoVar) /
              (51.0 * rhoVar);

        mxy = -(3.0 * mxxI * rhoI - 26.0 * mxyI * rhoI + 3.0 * myyI * rhoI +
                3.0 * rhoI * uxI - 3.0 * rhoI * uyI + rhoVar) /
              (17.0 * rhoVar);
    }
    else if (nodeTag == 3)
    {
        // const int incomings[Q] = {1, 1, 1, 1, 1, 1, 1, 0, 1};
        // const int outgoings[Q] = {1, 1, 1, 1, 1, 0, 1, 1, 1};

        const real omega = OMEGA;

        real rhoI = toReal(0.0);
        real uxI = toReal(0.0);
        real uyI = toReal(0.0);
        real mxxI = toReal(0.0);
        real myyI = toReal(0.0);
        real mxyI = toReal(0.0);
        for (int q = 0; q < Q; q++)
        {
            if (d_incomings_bcfluid[3][q] == 1)
            {
                const real cx = d_cx[q];
                const real cy = d_cy[q];

                const real Hxx = cx * cx - cs2;
                const real Hyy = cy * cy - cs2;
                const real Hxy = cx * cy;

                rhoI += pop[q];
                uxI += pop[q] * cx;
                uyI += pop[q] * cy;
                mxxI += pop[q] * Hxx;
                myyI += pop[q] * Hyy;
                mxyI += pop[q] * Hxy;
            }
        }
        if (rhoI <= 0.0)
            return;
        const real inv_rho = toReal(1.0) / rhoI;

        uxI *= inv_rho;
        uyI *= inv_rho;
        mxxI *= inv_rho;
        myyI *= inv_rho;
        mxyI *= inv_rho;

        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real linear_part = (5202.0 * rhoI + 612.0 * mxxI * rhoI + 1836.0 * mxyI * rhoI +
                                      612.0 * myyI * rhoI - 720.0 * mxxI * omega * rhoI -
                                      2160.0 * mxyI * omega * rhoI - 720.0 * myyI * omega * rhoI +
                                      255.0 * rhoI * uxI + 159.0 * omega * rhoI * uxI +
                                      255.0 * rhoI * uyI + 159.0 * omega * rhoI * uyI);

            const real inner_expr = (1734.0 + 204.0 * mxxI + 612.0 * mxyI + 204.0 * myyI -
                                     240.0 * mxxI * omega - 720.0 * mxyI * omega - 240.0 * myyI * omega +
                                     85.0 * uxI + 53.0 * omega * uxI + 85.0 * uyI + 53.0 * omega * uyI);

            const real squared_term = 3.0 * inner_expr * inner_expr;

            const real quadratic_terms = (45.0 * mxxI * mxxI + 405.0 * mxyI * mxyI + 45.0 * myyI * myyI -
                                          345.0 * myyI * uxI + 589.0 * uxI * uxI -
                                          345.0 * myyI * uyI + 1467.0 * uxI * uyI +
                                          589.0 * uyI * uyI +
                                          45.0 * mxyI * (6.0 * myyI - 23.0 * (uxI + uyI)) +
                                          15.0 * mxxI * (18.0 * mxyI + 6.0 * myyI - 23.0 * (uxI + uyI)));

            const real omega_factor = 2.0 * omega * (4998.0 + 103.0 * omega);

            const real sqrt_expr = sqrt(3.0 * rhoI * rhoI * (squared_term + omega_factor * quadratic_terms));

            rhoVar = (linear_part + sqrt_expr) / (9996.0 + 206.0 * omega);
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
            const real linear_part = 3.0 * rhoI * (1734.0 + 66.0 * mxxI + 198.0 * mxyI + 66.0 * myyI - 253.0 * uxI - 253.0 * uyI);
            const real inner_rho_sq = (31212.0 + 1602.0 * mxxI * mxxI + 14418.0 * mxyI * mxyI +
                                       2376.0 * myyI + 1602.0 * myyI * myyI - 9108.0 * uxI -
                                       12282.0 * myyI * uxI + 21041.0 * uxI * uxI +
                                       18.0 * mxyI * (396.0 + 534.0 * myyI - 2047.0 * uxI - 2047.0 * uyI) +
                                       6.0 * mxxI * (396.0 + 1602.0 * mxyI + 534.0 * myyI - 2047.0 * uxI - 2047.0 * uyI) -
                                       9108.0 * uyI - 12282.0 * myyI * uyI + 52080.0 * uxI * uyI +
                                       21041.0 * uyI * uyI);

            const real sqrt_part = 17.0 * SQRT_3 * std::sqrt(rhoI * rhoI * inner_rho_sq);

            rhoVar = (linear_part + sqrt_part) / 9998.0;
        }

        ux = -(3.0 * mxxI * rhoI + 9.0 * mxyI * rhoI + 3.0 * myyI * rhoI -
               20.0 * rhoI * uxI - 3.0 * rhoI * uyI + rhoVar) /
             (17.0 * rhoVar);

        uy = -(3.0 * mxxI * rhoI + 9.0 * mxyI * rhoI + 3.0 * myyI * rhoI -
               3.0 * rhoI * uxI - 20.0 * rhoI * uyI + rhoVar) /
             (17.0 * rhoVar);

        mxx = -(-57.0 * mxxI * rhoI - 18.0 * mxyI * rhoI - 6.0 * myyI * rhoI +
                6.0 * rhoI * uxI + 6.0 * rhoI * uyI - 2.0 * rhoVar) /
              (51.0 * rhoVar);

        myy = (6.0 * mxxI * rhoI + 18.0 * mxyI * rhoI + 57.0 * myyI * rhoI -
               6.0 * rhoI * uxI - 6.0 * rhoI * uyI + 2.0 * rhoVar) /
              (51.0 * rhoVar);

        mxy = (3.0 * mxxI * rhoI + 26.0 * mxyI * rhoI + 3.0 * myyI * rhoI -
               3.0 * rhoI * uxI - 3.0 * rhoI * uyI + rhoVar) /
              (17.0 * rhoVar);
    }
}

#endif // BOUDARIES_H