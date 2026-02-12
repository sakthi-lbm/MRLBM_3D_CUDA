#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include CASE_POST

#ifdef CYLINDER

inline void write_statistics(const nodeVar &fMom, const cylinderVar &cylinder, const int iter)
{
    cudaMemcpyFromSymbol(&h_TotalFx, d_TotalFx, sizeof(real));
    cudaMemcpyFromSymbol(&h_TotalFy, d_TotalFy, sizeof(real));
    cudaMemcpyFromSymbol(&h_Totalm, d_Totalm, sizeof(real));

    write_forces(iter);
    write_mass_flux(iter);

    calculate_write_inlet_average_density(fMom, iter);
    if (iter == STAT_START)
        calculate_write_theta(fMom, cylinder);
    calculate_write_pressure(fMom, cylinder, iter);
}

#endif

#endif