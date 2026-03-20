#include <iostream>
#include <fstream>
#include <numeric>

#include "main.cuh"

int main()
{
    //==================================== INITIALIZATION =====================================
    gpu_properties();
    create_output_directory();
    write_master_pvd();

    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    Simulation sim;
    setup_case(sim);
    initialize_simulation(sim);
    restart_simulation(sim);

    for (int iter = sim.start_iter; iter <= MAX_ITER; iter++)
    {
        streaming_and_evaluate_Mom<<<grid, block>>>(sim.d_caseData, sim.d_fMom, sim.d_fHalo, sim.d_gHalo, iter);
        checkKernelExecution();

        boundary_treatment(sim, iter);

        // post_streaming_pipeline();

        collision_halo_update<<<grid, block>>>(sim.d_fMom, sim.d_fHalo, sim.d_gHalo, iter);
        checkKernelExecution();
        swapHaloInterfaces(sim.d_fHalo, sim.d_gHalo);

        // post_collision_pipeline();

        post_step_pipeline(sim, iter);
    }

    calculate_mlups(sim.profile.sim_start_time, sim.profile.end_time, MAX_ITER, sim.profile.mlups);
    std::cout << "GLOBAL MLUPS: " << sim.profile.mlups << std::endl;

    finalize_simulation(sim);

    return 0;
}