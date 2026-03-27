#pragma once

#include "../../config.h"
#include STREAMING

// #include "nodeClass.cuh"

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
        return encodeNode(NODE_NORTH_WEST_FRONT, 0);
    if (isN && isW && isB)
        return encodeNode(NODE_NORTH_WEST_BACK, 0);
    if (isN && isE && isF)
        return encodeNode(NODE_NORTH_EAST_FRONT, 0);
    if (isN && isE && isB)
        return encodeNode(NODE_NORTH_EAST_BACK, 0);
    if (isS && isW && isF)
        return encodeNode(NODE_SOUTH_WEST_FRONT, 0);
    if (isS && isW && isB)
        return encodeNode(NODE_SOUTH_WEST_BACK, 0);
    if (isS && isE && isF)
        return encodeNode(NODE_SOUTH_EAST_FRONT, 0);
    if (isS && isE && isB)
        return encodeNode(NODE_SOUTH_EAST_BACK, 0);

    // --- EDGES
    if (isN && isW)
        return encodeNode(NODE_NORTH_WEST, 0);
    if (isN && isE)
        return encodeNode(NODE_NORTH_EAST, 0);
    if (isN && isF)
        return encodeNode(NODE_NORTH_FRONT, 0);
    if (isN && isB)
        return encodeNode(NODE_NORTH_BACK, 0);

    if (isS && isW)
        return encodeNode(NODE_SOUTH_WEST, 0);
    if (isS && isE)
        return encodeNode(NODE_SOUTH_EAST, 0);
    if (isS && isF)
        return encodeNode(NODE_SOUTH_FRONT, 0);
    if (isS && isB)
        return encodeNode(NODE_SOUTH_BACK, 0);

    if (isW && isF)
        return encodeNode(NODE_WEST_FRONT, 0);
    if (isW && isB)
        return encodeNode(NODE_WEST_BACK, 0);
    if (isE && isF)
        return encodeNode(NODE_EAST_FRONT, 0);
    if (isE && isB)
        return encodeNode(NODE_EAST_BACK, 0);

    // --- FACES
    if (isN)
        return encodeNode(NODE_NORTH, 0);
    if (isS)
        return encodeNode(NODE_SOUTH, 0);
    if (isW)
        return encodeNode(NODE_WEST, 0);
    if (isE)
        return encodeNode(NODE_EAST, 0);
    if (isF)
        return encodeNode(NODE_FRONT, 0);
    if (isB)
        return encodeNode(NODE_BACK, 0);

    // Default
    return encodeNode(NODE_BULK, 0);
}

__device__ inline void fluid_boundary_condition(const nodeType_t nodeTag, const uint32_t incomingMask,
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

    uint32_t incoming_mask = incomingMask;
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
    case TAG_BCF_1:
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
    case TAG_BCF_2:
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
    case TAG_BCF_3:
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
    case TAG_BCF_4:
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

__device__ inline void bcsolid_boundary_condition(const nodeType_t nodeTag,
                                                  const real *pop, real &rho, real &ux, real &uy, real &uz,
                                                  real &mxx, real &myy, real &mzz,
                                                  real &mxy, real &mxz, real &myz)
{
    switch (nodeTag)
    {
    case TAG_BCS_1:
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
    case TAG_BCS_2:
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
    case TAG_BCS_3:
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
    case TAG_BCS_4:
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
    case TAG_BCS_5:
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
    case TAG_BCS_6:
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
    case TAG_BCS_7:
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
    case TAG_BCS_8:
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
