#include <iostream>
#include <fstream>
#include <numeric>

#include "main.cuh"

int main()
{
    gpu_properties();
    create_output_directory();

    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    timestep sim_start_time = std::chrono::high_resolution_clock::now();
    timestep end_time;
    real mlups = 0.0;

    // variable declaration
    nodeVar h_fMom;
    nodeVar d_fMom;
    haloData fHalo_interface;
    haloData gHalo_interface;

    cylinderVar h_cylinder = {};
    cylinderVar d_cylinder = {};

    allocateHostMemory(h_fMom);
    allocateDeviceMemory(d_fMom);
    allocateHaloInterfaceMemory(fHalo_interface, gHalo_interface);

    initialize_host_device_constants();
    initialize_domain(d_fMom, h_fMom, gHalo_interface, h_cylinder, d_cylinder);

    return 0;
}