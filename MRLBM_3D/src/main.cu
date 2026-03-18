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
    timestep sim_start_time = std::chrono::high_resolution_clock::now();
    timestep step_start = std::chrono::high_resolution_clock::now();
    timestep step_end;
    timestep end_time;
    real mlups = 0.0;

    nodeVar h_fMom;
    nodeVar d_fMom;
    haloData h_fHalo;
    haloData fHalo_interface;
    haloData gHalo_interface;

    cylinderVar h_cylinder = {};
    cylinderVar d_cylinder = {};

    allocateHostMemory(h_fMom);
    allocateDeviceMemory(d_fMom);
    allocateHaloInterfaceMemory(h_fHalo, fHalo_interface, gHalo_interface);

    initialize_domain(d_fMom, h_fMom, gHalo_interface, h_cylinder, d_cylinder);

    copyMomentsDeviceToHost(h_fMom, d_fMom);
    copyNodeTypeHostToDevice(d_fMom, h_fMom);
    copyHaloInterfaces(fHalo_interface, gHalo_interface);

    writeSimInfo();
    checkCudaErrors(cudaMemcpyToSymbol(d_UCONV, &h_UCONV, sizeof(real)));

    //==================================== CHECKPOINT RESTART =====================================
    int start_iter = 0;
    if (RESTART)
    {
        start_iter = read_checkpoint(h_fMom, h_fHalo);
        std::cout << "Restarting from iteration " << start_iter << std::endl;
        copyMomentsHostToDevice(d_fMom, h_fMom);
        copyHaloHostToDevice(fHalo_interface, h_fHalo);
    }

    //==================================== MAIN LOOP =====================================
    for (int iter = start_iter; iter <= MAX_ITER; iter++)
    {
        //------------------------------------------Streaming ---------------------------------------------------------
        streaming_and_evaluate_Mom<<<grid, block>>>(d_cylinder, d_fMom, fHalo_interface, gHalo_interface, iter);
        checkKernelExecution();
        // compute_convective_outlet_velocity(d_fMom.ux);
#ifdef CYLINDER
        launch_incoming_force_kernal(d_fMom, d_cylinder, iter);

        constexpr dim3 boundary_block(BLOCK_NODES);
        const size_t CYLINDER_GRID_BLOCK = (NB + BLOCK_NODES - 1) / BLOCK_NODES;
        const dim3 boundary_grid(CYLINDER_GRID_BLOCK);
        apply_bc_cylinder<<<boundary_grid, boundary_block>>>(NB, INNER_NODE, d_cylinder, d_fMom,
                                                             UXP_CYLINDER, UYP_CYLINDER, UZP_CYLINDER,
                                                             D_WALL, iter);
        checkKernelExecution();
#endif

        //------------------------------------------Collision ---------------------------------------------------------
        collision_halo_update<<<grid, block>>>(d_cylinder, d_fMom, fHalo_interface, gHalo_interface, iter);
        checkKernelExecution();

        launch_outgoing_force_kernal(d_fMom, d_cylinder, iter);

        if (iter % MACR_SAVE == 0)
        {
            copyMomentsDeviceToHost(h_fMom, d_fMom);
            write_vti_3d(h_fMom, iter);

            printf("\n---------------------- (%d/%d) %.2f%% ----------------------\n", iter, MAX_ITER, toFloat(iter) / toFloat(MAX_ITER) * 100.0f);
            if (iter != 0)
                time_elapsing_count(step_end, step_start, iter);
            std::cout << "u_conv: " << h_UCONV << std::endl;
        }

        swapHaloInterfaces(fHalo_interface, gHalo_interface);

        //------------------------------------------Saving checkpoint ---------------------------------------------------------
        if (iter > start_iter && (iter % CHECKPOINT_SAVE == 0))
        {
            copyMomentsDeviceToHost(h_fMom, d_fMom);
            copyHaloDeviceToHost(h_fHalo, fHalo_interface);
            write_checkpoint(h_fMom, h_fHalo, iter + 1);
        }

        if (iter >= STAT_START && iter <= STAT_END)
        {
            write_statistics(h_fMom, h_cylinder, iter);
        }
    }
    //================================================ MAIN LOOP ENDS ================================================
    calculate_mlups(sim_start_time, end_time, MAX_ITER, mlups);
    std::cout << "GLOBAL MLUPS: " << mlups << std::endl;

    freeCylinderMemory(h_cylinder, d_cylinder);
    freeHostMemory(h_fMom);
    freeDeviceMemory(d_fMom);
    freeHaloInterfaceMemory(h_fHalo, fHalo_interface, gHalo_interface);

    return 0;
}