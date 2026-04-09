#include "simulation.cuh"
#include <numeric>

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

    copyMomentsDeviceToHost(sim.h_fMom, sim.d_fMom);

    initialize_nodeType(sim.h_fMom);

    sim.case_module.initialize(sim);

    copyNodeTypeHostToDevice(sim.d_fMom, sim.h_fMom);
    initialize_host_device_constants();
    
    copyHaloInterfaces(sim.d_fHalo, sim.d_gHalo);

    writeSimInfo();
}

void compute_active_blocks(Simulation &sim)
{
    sim.h_active_blocks.resize(TOTAL_BLOCKS, 0);

    checkCudaErrors(cudaMalloc(&sim.d_active_blocks, TOTAL_BLOCKS * sizeof(int)));

    find_active_blocks_3d(sim.h_fMom, sim.h_active_blocks);

    checkCudaErrors(cudaMemcpy(sim.d_active_blocks,
                               sim.h_active_blocks.data(),
                               TOTAL_BLOCKS * sizeof(int),
                               cudaMemcpyHostToDevice));

    sim.total_active_blocks = std::accumulate(sim.h_active_blocks.begin(),
                                              sim.h_active_blocks.end(), 0);

    printf("Total Blocks: %zu\n", (size_t)TOTAL_BLOCKS);

    printf("Active Blocks: %d (%.2f%%)\n",
           sim.total_active_blocks,
           (sim.total_active_blocks * 100.0f) / TOTAL_BLOCKS);

    printf("Skipped Blocks: %zu\n",
           (size_t)(TOTAL_BLOCKS - sim.total_active_blocks));
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
    streaming_and_evaluate_Mom<<<grid, block>>>(sim.d_caseData, sim.d_fMom, sim.d_fHalo, sim.d_gHalo, sim.d_active_blocks, iter);
    checkKernelExecution();

    compute_convective_outlet_velocity(sim.d_fMom.ux);
}

void collision(Simulation &sim, int iter)
{
    collision_halo_update<<<grid, block>>>(sim.d_fMom, sim.d_fHalo, sim.d_gHalo, sim.d_active_blocks, iter);
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
    if (sim.case_module.post_streaming)
        sim.case_module.post_streaming(sim, iter);
}

void post_collision_pipeline(Simulation &sim, int iter)
{
    if (sim.case_module.post_collision)
        sim.case_module.post_collision(sim, iter);
}

void post_step_pipeline(Simulation &sim, int iter)
{
    handle_checkpoint(sim, iter);
    handle_output(sim, iter);
    handle_statistics(sim, iter);
}

void handle_statistics(Simulation &sim, int iter)
{
    if (iter >= STAT_START && iter <= STAT_END)
    {
        if (sim.case_module.post_process_step)
            sim.case_module.post_process_step(sim, iter);
    }
}

void handle_output(Simulation &sim, int iter)
{
    if (iter % MACR_SAVE == 0)
    {
        copyMomentsDeviceToHost(sim.h_fMom, sim.d_fMom);

        if (sim.case_module.write_output)
            sim.case_module.write_output(sim, iter);

        printf("\n---------------------- (%d/%d) %.2f%% ----------------------\n", iter, MAX_ITER, toFloat(iter) / toFloat(MAX_ITER) * 100.0f);
        time_elapsing_count(sim.profile.sim_start_time, sim.profile.step_end, sim.profile.step_start,
                            iter, sim.start_iter);
    }
}

void handle_checkpoint(Simulation &sim, int iter)
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

void find_active_blocks_3d(const nodeVar &fMom, std::vector<int> &h_active_blocks)
{
    for (size_t bz = 0; bz < GRID_BLOCK_Z; ++bz)
    {
        for (size_t by = 0; by < GRID_BLOCK_Y; ++by)
        {
            for (size_t bx = 0; bx < GRID_BLOCK_X; ++bx)
            {
                bool has_fluid = false;

                for (size_t tz = 0; tz < BLOCK_THREAD_Z && !has_fluid; ++tz)
                {
                    for (size_t ty = 0; ty < BLOCK_THREAD_Y && !has_fluid; ++ty)
                    {
                        for (size_t tx = 0; tx < BLOCK_THREAD_X; ++tx)
                        {
                            const unsigned int x = tx + bx * BLOCK_THREAD_X;
                            const unsigned int y = ty + by * BLOCK_THREAD_Y;
                            const unsigned int z = tz + bz * BLOCK_THREAD_Z;

                            // global bounds check
                            if (x >= NX || y >= NY || z >= NZ)
                                continue;

                            // IMPORTANT: ensure this matches your 3D layout
                            const size_t idx = IDX_BLOCK(tx, ty, tz, bx, by, bz);

                            if (getType(fMom.nodeType[idx]) != NODE_SOLID)
                            {
                                has_fluid = true;
                                break;
                            }
                        }
                    }
                }
                const size_t blockIdx = bx + by * GRID_BLOCK_X + bz * GRID_BLOCK_X * GRID_BLOCK_Y;
                h_active_blocks[blockIdx] = has_fluid ? 1 : 0;
            }
        }
    }
}