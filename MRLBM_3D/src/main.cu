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
    timestep end_time;
    real mlups = 0.0;

    // variable declaration
    nodeVar h_fMom;
    nodeVar d_fMom;
    cylinderVar h_cylinder = {};
    cylinderVar d_cylinder = {};

    allocateHostMemory(h_fMom);
    allocateDeviceMemory(d_fMom);
    initialize_host_device_constants();
    initialize_domain(d_fMom, h_fMom, h_cylinder, d_cylinder);

    copyMomentsDeviceToHost(h_fMom, d_fMom);
    copyNodeTypeHostToDevice(d_fMom, h_fMom);

    writeSimInfo();
    for (int iter = 0; iter <= 0; iter++)
    {
        streaming_and_evaluate_Mom<<<grid, block>>>(d_cylinder, d_fMom, iter);
        checkKernelExecution();

        // collision_halo_update<<<grid, block>>>(d_cylinder, d_fMom, iter);
        checkKernelExecution();

        if (iter % MACR_SAVE == 0)
        {
            copyMomentsDeviceToHost(h_fMom, d_fMom);
            write_vti_3d(h_fMom, iter);

            printf("\n---------------------- (%d/%d) %.2f%% ----------------------\n", iter, MAX_ITER, toFloat(iter) / toFloat(MAX_ITER) * 100.0f);
        }
    }

    calculate_mlups(sim_start_time, end_time, MAX_ITER, mlups);
    std::cout << "GLOBAL MLUPS: " << mlups << std::endl;

    return 0;
}