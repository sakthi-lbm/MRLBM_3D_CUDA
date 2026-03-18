#ifndef STAT_HEADER_H
#define STAT_HEADER_H

#include "../solver/initializeLBM_inline.cuh"

void launch_incoming_force_kernal(const nodeVar &dMom, const cylinderVar &cylinder, const int iter);
void launch_outgoing_force_kernal(const nodeVar &dMom, const cylinderVar &cylinder, const int iter);

__global__ void compute_incoming_force_mass_kernal(const nodeVar dMom, cylinderVar cylinder);
__global__ void compute_outgoing_force_mass_kernal(const nodeVar dMom, cylinderVar cylinder);



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