#ifndef SAVE_DATA_H
#define SAVE_DATA_H

#include <vector>
#include "../restart/restart_io.cuh"

void write_master_pvd();
void write_vti_3d(nodeVar data, int timestep);

inline void write_statistics(const nodeVar &fMom, const cylinderVar &cylinder, const int iter)
{
    // cudaMemcpyFromSymbol(&h_TotalFx, d_TotalFx, sizeof(real));
    // cudaMemcpyFromSymbol(&h_TotalFy, d_TotalFy, sizeof(real));
    // cudaMemcpyFromSymbol(&h_TotalFz, d_TotalFz, sizeof(real));
    // cudaMemcpyFromSymbol(&h_Totalm, d_Totalm, sizeof(real));

    // write_forces(iter);
    // write_mass_flux(iter);
}

#endif // SAVE_DATA_H