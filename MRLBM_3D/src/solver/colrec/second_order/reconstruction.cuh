#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "../../../config.h"



__device__ inline void pop_reconstruction(const real rhoVar, const real uxVar, const real uyVar, const real mxxVar, const real myyVar, const real mxyVar, real *pop)
{
    // for (size_t i = 0; i < Q; i++)
    // {
    //     const real Hxx = d_cx[i] * d_cx[i] - cs2;
    //     const real Hyy = d_cy[i] * d_cy[i] - cs2;
    //     const real Hxy = d_cx[i] * d_cy[i];

    //     pop[i] = w[i] * rho * (toReal(1.0) + as2 * (ux * d_cx[i] + uy * d_cy[i]) + toReal(0.5) * as2 * as2 * (Hxx * mxx + Hyy * myy + toReal(2.0) * Hxy * mxy));
    // }

    const real rho = rhoVar * F_M_0_SCALE;
    const real ux = uxVar * F_M_I_SCALE;
    const real uy = uyVar * F_M_I_SCALE;
    const real mxx = mxxVar * F_M_II_SCALE;
    const real myy = myyVar * F_M_II_SCALE;
    const real mxy = mxyVar * F_M_IJ_SCALE;

    real pics2 = toReal(1.0) - cs2 * (mxx + myy);

    real multiplyTerm = W0 * (rho);
    pop[0] = multiplyTerm * (pics2);

    multiplyTerm = W1 * (rho);
    pop[1] = multiplyTerm * (pics2 + ux + mxx);
    pop[2] = multiplyTerm * (pics2 + uy + myy);
    pop[3] = multiplyTerm * (pics2 - ux + mxx);
    pop[4] = multiplyTerm * (pics2 - uy + myy);

    multiplyTerm = W2 * (rho);
    pop[5] = multiplyTerm * (pics2 + ux + uy + mxx + myy + mxy);
    pop[6] = multiplyTerm * (pics2 - ux + uy + mxx + myy - mxy);
    pop[7] = multiplyTerm * (pics2 - ux - uy + mxx + myy + mxy);
    pop[8] = multiplyTerm * (pics2 + ux - uy + mxx + myy - mxy);
}

#endif