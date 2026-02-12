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
    for (size_t q = 0; q < Q; q++)
    {
        h_Hxx[q] = h_cx[q] * h_cx[q] - cs2;
        h_Hyy[q] = h_cy[q] * h_cy[q] - cs2;
        h_Hzz[q] = h_cz[q] * h_cz[q] - cs2;
        h_Hxy[q] = h_cx[q] * h_cy[q];
        h_Hxz[q] = h_cx[q] * h_cz[q];
        h_Hyz[q] = h_cy[q] * h_cz[q];
    }
    // copy to GPU constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_w, &h_w, sizeof(h_w)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hxx, &h_Hxx, sizeof(h_Hxx)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hyy, &h_Hyy, sizeof(h_Hyy)));
    checkCudaErrors(cudaMemcpyToSymbol(d_Hxy, &h_Hxy, sizeof(h_Hxy)));

    check_tau();
    check_mach();
}

#endif // INITIALIZE_LBM_INLINE_H
