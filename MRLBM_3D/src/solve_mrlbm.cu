#include "solve_mrlbm.cuh"

void setup_case(Simulation &sim)
{

// Case kernal selection
#if BC_PROBLEM == cylinder

    setup_cylinder_case(sim.case_module);

#elif BC_PROBLEM == airfoil

    setup_airfoil_case(sim.case_module);

#elif BC_PROBLEM == annulus

    setup_annulus_case(sim.case_module);

#else

#error "Unknown BC_PROBLEM"

#endif
}