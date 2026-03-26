#pragma once

#include <fstream>
#include <iostream>
#include "cudaHelpers.cuh"

inline void create_output_directory()
{
#if defined(_WIN32)
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "\\\\"; // adds "\\"
    strPath += ID_SIM;
    std::string cmd = "md ";
    cmd += strPath;
    system(cmd.c_str());
    return;
#endif // !_WIN32

#if defined(__APPLE__) || defined(__MACH__) || defined(__linux__)
    std::string Path;
    Path = construct_path(PATH_FILES, ID_SIM, "plots");
    std::string cmd = "mkdir -p ";
    cmd += Path;
    const int i = system(cmd.c_str());
    static_cast<void>(i);

    Path = construct_path(PATH_FILES, ID_SIM, "grid_layout");
    cmd = "mkdir -p ";
    cmd += Path;
    const int j = system(cmd.c_str());
    static_cast<void>(j);

    Path = construct_path(PATH_FILES, ID_SIM, "checkpoint");
    cmd = "mkdir -p ";
    cmd += Path;
    const int k = system(cmd.c_str());
    static_cast<void>(k);

    return;

#endif
    printf("I don't know how to setup folders for your operational system :(\n");
    return;
}

inline std::string getSimInfoString()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, GPU_INDEX);

    std::string precision;
    if (typeid(real) == typeid(double))
        precision = "double";
    else if (typeid(real) == typeid(float))
        precision = "float";
    else
        precision = "unknown";

    std::ostringstream out;
    out << std::fixed << std::setprecision(6);

    const int labelWidth = 25;

    // Physical units
    const real L_phy = 1.0;
    const real nu_phy = 0.01;
    const real u_phy = RE * nu_phy / L_phy;

    const real delx_phy = L_phy / D_OUT;
    const real delt_phy = (delx_phy * delx_phy) * VISC / nu_phy;

    out << "========================= SIMULATION INFORMATION =========================\n";
    out << std::left;
    out << std::setw(labelWidth) << "Simulation ID" << " : " << ID_SIM << "\n";
    out << std::setw(labelWidth) << "Velocity set" << " : D3Q" << Q << "\n";
    out << std::setw(labelWidth) << "Re" << " : " << toInt(RE) << "\n";
    out << std::setw(labelWidth) << "Precision" << " : " << precision << "\n";
    out << std::setw(labelWidth) << std::left << "Grid size" << " : " << NX << " x "
        << NY << " x " << NZ << "\n";
    out << std::setw(labelWidth) << "Total Grid points" << " : " << NX * NY << "\n";
    out << std::setw(labelWidth) << "Grid points (in million)" << " : " << toReal(NX * NY) / 1000000.0 << "\n";
    out << "\n";

    out << "----------------------------- CYLINDER properties -----------------------------\n";
    out << std::setw(labelWidth) << "Stair-case" << " : " << !triangular << "\n";
    out << std::setw(labelWidth) << "Triangular" << " : " << triangular << "\n";
    out << std::setw(labelWidth) << "Diameter" << " : " << D_IN << "\n";
    out << std::setw(labelWidth) << "Radius" << " : " << R_IN << "\n";
    out << std::setw(labelWidth) << "Center xc" << " : " << XC << "\n";
    out << std::setw(labelWidth) << "Center yc" << " : " << YC << "\n";

    out << "\n";

    out << "----------------------------- Lattice Units -----------------------------\n";
    out << std::setw(labelWidth) << "uo" << " : " << U_MAX << "\n";
    out << std::setw(labelWidth) << "Mach" << " : " << U_MAX / sqrt(cs2) << "\n";
    out << std::setw(labelWidth) << "Viscosity" << " : " << VISC << "\n";
    out << std::setw(labelWidth) << "Tau" << " : " << TAU << "\n";
    out << std::setw(labelWidth) << "Omega" << " : " << OMEGA << "\n";
    out << std::setw(labelWidth) << "Macr_save" << " : " << MACR_SAVE << "\n";
    out << std::setw(labelWidth) << "Nsteps" << " : " << MAX_ITER << "\n";

    out << "\n";
    out << "----------------------------- Physical Units -----------------------------\n";
    out << std::setw(labelWidth) << "L_phy" << " : " << L_phy << "\n";
    out << std::setw(labelWidth) << "dx_phy" << " : " << delx_phy << "\n";
    out << std::setw(labelWidth) << "dt_phy" << " : " << delt_phy << "\n";
    out << std::setw(labelWidth) << "u_phy" << " : " << u_phy << "\n";
    out << std::setw(labelWidth) << "nu_phy" << " : " << nu_phy << "\n";
    out << "\n";
    out << "----------------------------- CUDA Parameters -----------------------------\n";
    out << std::setw(labelWidth) << std::left << "Block size" << " : " << BLOCK_THREAD_X << " x "
        << BLOCK_THREAD_Y << " x "
        << BLOCK_THREAD_Z << "\n";
    out << std::setw(labelWidth) << "Total threads per block" << " : " << THREADS_PER_BLOCK << "\n";
    out << std::setw(labelWidth) << std::left << "Grid size" << " : " << GRID_BLOCK_X << " x "
        << GRID_BLOCK_Y << " x "
        << GRID_BLOCK_Z << "\n";
    out << std::setw(labelWidth) << "Total blocks"
        << " : " << NUMBER_OF_BLOCKS << "\n";

    out << std::setw(labelWidth) << "Max Shared Memory (KB)"
        << " : " << MAX_SHARED_MEM_BYTES / BYTES_PER_KB << "\n";

    out << std::setw(labelWidth) << "Shared Memory used (KB)"
        << " : " << static_cast<double>(USED_SHARED_MEMORY_BYTES) / BYTES_PER_KB << "\n";

    out << std::setw(labelWidth) << "Total Global Memory (MB)"
        << " : " << static_cast<double>(prop.totalGlobalMem) / BYTES_PER_MB << "\n";

    out << std::setw(labelWidth) << "Moments Memory used (MB)"
        << " : " << static_cast<double>(USED_GLOBAL_MEMORY_BYTES) / BYTES_PER_MB << "\n";

    out << std::setw(labelWidth) << "Halo Memory used (MB)"
        << " : " << static_cast<double>(HALO_GLOBAL_MEMORY_BYTES) / BYTES_PER_MB << "\n";
    out << "==========================================================================\n";

    return out.str();
}

inline void writeSimInfo()
{
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "/";
    strPath += ID_SIM;
    strPath += "/";
    strPath += ID_SIM;
    strPath += "_Sim_info.txt";

    std::string info = getSimInfoString();

    // Print to console
    std::cout << info;

    // Write to file
    std::ofstream outFile(strPath, std::ios::out);
    if (!outFile)
    {
        std::cerr << "Error: Could not open file " << strPath << " for writing simulation info.\n";
        return;
    }

    outFile << info;
    outFile.close();
}

#ifdef CYLINDER

inline void write_postprocess_info(const real mlups)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "postprocess_info.txt");
    std::ofstream out(filename);

    if (!out)
    {
        throw std::runtime_error("Cannot open postprocess file: " + filename);
    }

    out << std::fixed << std::setprecision(6);
    out << "MLUPS       = " << mlups << "\n";
    out << "D           = " << D << "\n";
    out << "D_max       = " << D_WALL << "\n";
    // out << "NB          = " << NB << "\n";
    out << "Re          = " << RE << "\n";
    out << "uo          = " << U_MAX << "\n";
    out << "rho_infty   = " << rho_infty << "\n";
}
#endif

inline void calculate_mlups(timestep &tstart, timestep &tend, int steps, real &mlups)
{
    tend = std::chrono::high_resolution_clock::now();
    double step_time = std::chrono::duration<double>(tend - tstart).count();
    if (step_time > 0.0)
        mlups = (static_cast<double>(NUM_LBM_NODES) * steps / 1e6) / step_time;
    else
        mlups = 0.0;

    tstart = std::chrono::high_resolution_clock::now();
}

inline void gpu_properties()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "Number of CUDA devices: " << device_count << std::endl;

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << std::endl;
    }
}

__host__ inline void time_elapsing_count(timestep &step_end, timestep &step_start, size_t step, int start_step)
{
    step_end = std::chrono::high_resolution_clock::now();
    double step_time = std::chrono::duration<double>(step_end - step_start).count();

    // Calculate MLUPS for the current step
    real MLUPS = (NUM_LBM_NODES * MACR_SAVE / 1e6) / step_time;

    // Calculate remaining time
    size_t steps_remaining = MAX_ITER - step;
    double total_seconds_remaining = steps_remaining * NUM_LBM_NODES / 1e6 / MLUPS;

    // Convert to hours, minutes, seconds
    size_t hours = static_cast<size_t>(total_seconds_remaining) / 3600;
    size_t minutes = static_cast<size_t>((total_seconds_remaining - hours * 3600) / 60);
    size_t seconds = static_cast<size_t>(total_seconds_remaining) % 60;

    if (step > start_step)
    {
        std::cout << "Elapsed time: " << step_time << " seconds" << std::endl;
        std::cout << "MLUPS: " << MLUPS << std::endl;

        std::cout << "Estimated time left: "
                  << hours << "h "
                  << minutes << "min "
                  << seconds << "s" << std::endl;
    }
    else
    {
        std::cout << "Elapsed time: -- " << std::endl;
        std::cout << "MLUPS: -- " << std::endl;
        std::cout << "Estimated time left: -- " << std::endl;
    }

    step_start = std::chrono::high_resolution_clock::now();
}
