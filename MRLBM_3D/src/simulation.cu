#include "simulation.cuh"

void setup_environment()
{
    gpu_properties();
    create_output_directory();
    write_master_pvd();

    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    
}

void initialize_simulation(Simulation &sim)
{
    sim.profile.sim_start_time = std::chrono::high_resolution_clock::now();
    sim.profile.step_start = std::chrono::high_resolution_clock::now();

    allocateHostMemory(sim.h_fMom);
    allocateDeviceMemory(sim.d_fMom);
    allocateHaloInterfaceMemory(sim.h_fHalo, sim.d_fHalo, sim.d_gHalo);

    gpu_initialize_Moments_GhostInterface<<<grid, block>>>(sim.d_fMom, sim.d_gHalo);
    checkKernelExecution();

    initialize_nodeType(sim.h_fMom);

    sim.case_module.initialize(sim);

    copyNodeTypeHostToDevice(sim.d_fMom, sim.h_fMom);
    initialize_host_device_constants();
    copyMomentsDeviceToHost(sim.h_fMom, sim.d_fMom);
    copyHaloInterfaces(sim.d_fHalo, sim.d_gHalo);
    writeSimInfo();
}

void restart_simulation(Simulation &sim)
{
    if constexpr (RESTART)
    {
        sim.start_iter = read_checkpoint(sim.h_fMom, sim.h_fHalo);

        std::cout << "Restarting from iteration "
                  << sim.start_iter << std::endl;

        copyMomentsHostToDevice(sim.d_fMom, sim.h_fMom);
        copyHaloHostToDevice(sim.d_fHalo, sim.h_fHalo);
    }
    else
    {
        sim.start_iter = 0;
    }
}

void streaming(Simulation &sim, int iter)
{
    streaming_and_evaluate_Mom<<<grid, block>>>(sim.d_caseData, sim.d_fMom, sim.d_fHalo, sim.d_gHalo, iter);
    checkKernelExecution();

    compute_convective_outlet_velocity(sim.d_fMom.ux);
}

void collision(Simulation &sim, int iter)
{
    collision_halo_update<<<grid, block>>>(sim.d_fMom, sim.d_fHalo, sim.d_gHalo, iter);
    checkKernelExecution();

    swapHaloInterfaces(sim.d_fHalo, sim.d_gHalo);
}

void boundary_treatment(Simulation &sim, int iter)
{
    if (sim.case_module.apply_boundary)
        sim.case_module.apply_boundary(sim, iter);
}

void post_streaming_pipeline(Simulation &sim, int iter)
{
    // if (sim.case_module.compute_forces)
    //     sim.case_module.compute_forces(sim, iter);
}

void post_collision_pipeline(Simulation &sim, int iter)
{
    // if (sim.case_module.compute_forces)
    //     sim.case_module.compute_forces(sim, iter);
}

void post_step_pipeline(Simulation &sim, int iter)
{
    save_checkpoint(sim, iter);

    if (iter % MACR_SAVE == 0)
    {
        copyMomentsDeviceToHost(sim.h_fMom, sim.d_fMom);
        write_vti_3d(sim.h_fMom, iter);

        printf("\n---------------------- (%d/%d) %.2f%% ----------------------\n", iter, MAX_ITER, toFloat(iter) / toFloat(MAX_ITER) * 100.0f);
        time_elapsing_count(sim.profile.step_end, sim.profile.step_start, iter, sim.start_iter);
    }

    if (iter >= STAT_START && iter <= STAT_END)
    {
        if (sim.case_module.post_process_step)
            sim.case_module.post_process_step(sim, iter);
    }
}

void save_checkpoint(Simulation &sim, int iter)
{
    if (iter > sim.start_iter && (iter % CHECKPOINT_SAVE == 0))
    {
        // Copy data back to host
        copyMomentsDeviceToHost(sim.h_fMom, sim.d_fMom);
        copyHaloDeviceToHost(sim.h_fHalo, sim.d_fHalo);

        // Write checkpoint
        write_checkpoint(sim.h_fMom, sim.h_fHalo, iter + 1);

        std::cout << "Checkpoint saved at iteration " << iter << std::endl;
    }
}

void finalize_simulation(Simulation &sim)
{
    calculate_mlups(sim.profile.sim_start_time, sim.profile.end_time, MAX_ITER, sim.profile.mlups);
    std::cout << "GLOBAL MLUPS: " << sim.profile.mlups << std::endl;

    if (sim.case_module.free_case)
        sim.case_module.free_case(sim);

    freeHostMemory(sim.h_fMom);
    freeDeviceMemory(sim.d_fMom);
    freeHaloInterfaceMemory(sim.h_fHalo, sim.d_fHalo, sim.d_gHalo);
}