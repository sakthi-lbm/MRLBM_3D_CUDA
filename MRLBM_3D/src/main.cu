#include <iostream>
#include <fstream>
#include <numeric>

#include "main.cuh"

int main()
{
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
    haloData fHalo_interface;
    haloData gHalo_interface;

    cylinderVar h_cylinder = {};
    cylinderVar d_cylinder = {};

    allocateHostMemory(h_fMom);
    allocateDeviceMemory(d_fMom);
    allocateHaloInterfaceMemory(fHalo_interface, gHalo_interface);

    initialize_domain(d_fMom, h_fMom, gHalo_interface, h_cylinder, d_cylinder);

    copyMomentsDeviceToHost(h_fMom, d_fMom);
    copyNodeTypeHostToDevice(d_fMom, h_fMom);
    copyHaloInterfaces(fHalo_interface, gHalo_interface);

    writeSimInfo();

    for (int iter = 0; iter <= MAX_ITER; iter++)
    {
        streaming_and_evaluate_Mom<<<grid, block>>>(d_cylinder, d_fMom, fHalo_interface, gHalo_interface, iter);
        checkKernelExecution();

#ifdef CYLINDER
        constexpr size_t CYLINDER_NODES = 256;
        constexpr dim3 boundary_block(CYLINDER_NODES);

        const size_t CYLINDER_GRID_BLOCK = (NB + CYLINDER_NODES - 1) / CYLINDER_NODES;
        const dim3 boundary_grid(CYLINDER_GRID_BLOCK);

        apply_bc_cylinder<<<boundary_grid, boundary_block>>>(NB, INNER_NODE, d_cylinder, d_fMom,
                                                             UXP_CYLINDER, UYP_CYLINDER, UZP_CYLINDER,
                                                             D_WALL, iter);
        checkKernelExecution();
#endif
        if (iter % MACR_SAVE == 0)
        {
            copyMomentsDeviceToHost(h_fMom, d_fMom);
            write_vti_3d(h_fMom, iter);

            printf("\n---------------------- (%d/%d) %.2f%% ----------------------\n", iter, MAX_ITER, toFloat(iter) / toFloat(MAX_ITER) * 100.0f);
            if (iter != 0)
                time_elapsing_count(step_end, step_start, iter);
        }

        collision_halo_update<<<grid, block>>>(d_cylinder, d_fMom, fHalo_interface, gHalo_interface, iter);
        checkKernelExecution();

        swapHaloInterfaces(fHalo_interface, gHalo_interface);
    }

    calculate_mlups(sim_start_time, end_time, MAX_ITER, mlups);
    std::cout << "GLOBAL MLUPS: " << mlups << std::endl;

    freeCylinderMemory(h_cylinder, d_cylinder);
    freeHostMemory(h_fMom);
    freeDeviceMemory(d_fMom);
    freeHaloInterfaceMemory(fHalo_interface, gHalo_interface);

    return 0;
}