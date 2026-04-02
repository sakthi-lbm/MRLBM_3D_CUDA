#ifndef STAT_HEADER_H
#define STAT_HEADER_H

#include "../config.h"

#include CASE_BOUNDARY

__global__ void compute_force_mass_kernel(const nodeVar dMom,
                                          const size_t *boundaryList,
                                          const size_t *bcfluidList,
                                          const uint32_t *boundaryMask,
                                          const int NB, const int NB_FLUID,
                                          const MaskType masktype,
                                          const real sign);

inline void write_forces(const int iter)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "forces.dat");
    static bool first_call = true;
    if (first_call)
    {
        std::ofstream clear(filename, std::ios::trunc); // delete contents
        first_call = false;
    }

    std::ofstream forcefile(filename, std::ios::app);

    if (forcefile.is_open())
    {
        forcefile << std::setprecision(16) << std::fixed
                  << iter << " " << h_TotalFx << " " << h_TotalFy << " " << h_TotalFz << std::endl;
    }
}

inline void write_mass_flux(const int iter)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "mass_flux.dat");
    static bool first_call = true;
    if (first_call)
    {
        std::ofstream clear(filename, std::ios::trunc); // delete contents
        first_call = false;
    }

    std::ofstream massfile(filename, std::ios::app);

    if (massfile.is_open())
    {
        massfile << std::setprecision(16) << std::fixed << iter << " " << h_Totalm << std::endl;
    }
}

#endif // STAT_HEADER_H