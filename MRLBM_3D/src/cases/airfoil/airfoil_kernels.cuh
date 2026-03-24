#pragma once

#include "airfoil_helpers.cuh"

void airfoil_initialize(Simulation &sim);
void airfoil_apply_boundary(Simulation &sim, int iter);
void airfoil_post_streaming(Simulation &sim, int iter);
void airfoil_post_collision(Simulation &sim, int iter);
void setup_airfoil_case(Case &case_module);

void airfoil_incoming_force_kernal(Simulation &sim, const int iter);
void airfoil_outgoing_force_kernal(Simulation &sim, const int iter);

void airfoil_post_process(Simulation &sim, int iter);

__device__ void airfoil_boundary_moments(nodeType_t nodeType, airfoilVar &airfoil, nodeVar &dMom, real *pop,
                                          real &rho, real &ux, real &uy, real &uz,
                                          real &mxx, real &myy, real &mzz,
                                          real &mxy, real &mxz, real &myz);

__global__ void apply_bc_airfoil(const int NB, const nodeType_t NODE_TYPE, const airfoilVar &airfoil,
                                  nodeVar dMom, const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                  const real D_WALL, const int iter);

__global__ void compute_surface_pressure(const nodeVar &dMom, const airfoilVar &d_airfoil,
                                         const airfoilPostProcess &d_airfoilPost,
                                         const int n_avg);

//=================================================================================================================

inline void allocateairfoilMemory(airfoilVar &h_airfoil, airfoilVar &d_airfoil,
                                   airfoilPostProcess &h_airfoilPost, airfoilPostProcess &d_airfoilPost)
{
    const int NB = h_airfoil.NB;
    const int NB_FLUID = h_airfoil.NB_FLUID;
    const int NB_SOLID = h_airfoil.NB_SOLID;

    if (NB <= 0)
    {
        std::cerr << "Error: NB not initialized!\n";
        exit(EXIT_FAILURE);
    }

    // ================= HOST =================
    checkCudaErrors(cudaMallocHost(&h_airfoil.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_airfoil.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMallocHost(&h_airfoil.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMallocHost(&h_airfoil.bcfluidList, NB_FLUID * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_airfoil.bcsolidList, NB_SOLID * sizeof(size_t)));

    checkCudaErrors(cudaMallocHost(&h_airfoil.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_airfoil.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_airfoil.delta_w, NB * sizeof(real)));

    checkCudaErrors(cudaMallocHost(&h_airfoilPost.Cp_avg, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_airfoilPost.Cp_rms_avg, NB * sizeof(real)));
    // checkCudaErrors(cudaMallocHost(&h_airfoilPost.Cp_span_avg, NB * sizeof(real)));

    memset(h_airfoilPost.Cp_avg, 0, NB * sizeof(real));
    memset(h_airfoilPost.Cp_rms_avg, 0, NB * sizeof(real));
    // memset(h_airfoilPost.Cp_span_avg, 0, NB * sizeof(real));
    h_airfoilPost.n_avg = 0;

    // ================= DEVICE =================
    checkCudaErrors(cudaMalloc(&d_airfoil.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_airfoil.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&d_airfoil.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMalloc(&d_airfoil.bcfluidList, NB_FLUID * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_airfoil.bcsolidList, NB_SOLID * sizeof(size_t)));

    checkCudaErrors(cudaMalloc(&d_airfoil.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_airfoil.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_airfoil.delta_w, NB * sizeof(real)));

    checkCudaErrors(cudaMallocHost(&d_airfoilPost.Cp_avg, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&d_airfoilPost.Cp_rms_avg, NB * sizeof(real)));
    // checkCudaErrors(cudaMallocHost(&d_airfoilPost.Cp_span_avg, NB * sizeof(real)));

    cudaMemset(d_airfoilPost.Cp_avg, 0, NB * sizeof(real));
    cudaMemset(d_airfoilPost.Cp_rms_avg, 0, NB * sizeof(real));
    // cudaMemset(d_airfoilPost.Cp_span_avg, 0, NB * sizeof(real));
}

inline void freeairfoilMemory(airfoilVar &h_airfoil, airfoilVar &d_airfoil)
{
    // Free-ing host memory
    checkCudaErrors(cudaFreeHost(h_airfoil.boundaryList));
    checkCudaErrors(cudaFreeHost(h_airfoil.incomingMask));
    checkCudaErrors(cudaFreeHost(h_airfoil.outgoingMask));

    checkCudaErrors(cudaFreeHost(h_airfoil.bcfluidList));
    checkCudaErrors(cudaFreeHost(h_airfoil.bcsolidList));

    // Free-ing device memory
    checkCudaErrors(cudaFree(d_airfoil.boundaryList));
    checkCudaErrors(cudaFree(d_airfoil.incomingMask));
    checkCudaErrors(cudaFree(d_airfoil.outgoingMask));

    checkCudaErrors(cudaFree(d_airfoil.bcfluidList));
    checkCudaErrors(cudaFree(d_airfoil.bcsolidList));
}

inline void airfoil_free(Simulation &sim)
{
    auto *h_cyl = static_cast<airfoilVar *>(sim.h_caseData);
    auto *d_cyl = static_cast<airfoilVar *>(sim.d_caseData);
    auto *h_cylPost = static_cast<airfoilPostProcess *>(sim.h_casePost);
    auto *d_cylPost = static_cast<airfoilPostProcess *>(sim.d_casePost);

    if (!h_cyl || !d_cyl)
        return;

    // Free host memory
    cudaFreeHost(h_cyl->boundaryList);
    cudaFreeHost(h_cyl->incomingMask);
    cudaFreeHost(h_cyl->outgoingMask);

    cudaFreeHost(h_cyl->bcfluidList);
    cudaFreeHost(h_cyl->bcsolidList);

    cudaFreeHost(h_cyl->unit_nx);
    cudaFreeHost(h_cyl->unit_ny);
    cudaFreeHost(h_cyl->delta_w);

    cudaFreeHost(h_cylPost->Cp_avg);
    cudaFreeHost(h_cylPost->Cp_rms_avg);

    // Free device memory
    cudaFree(d_cyl->boundaryList);
    cudaFree(d_cyl->incomingMask);
    cudaFree(d_cyl->outgoingMask);

    cudaFree(d_cyl->bcfluidList);
    cudaFree(d_cyl->bcsolidList);

    cudaFree(d_cyl->unit_nx);
    cudaFree(d_cyl->unit_ny);
    cudaFree(d_cyl->delta_w);

    cudaFreeHost(d_cylPost->Cp_avg);
    cudaFreeHost(d_cylPost->Cp_rms_avg);

    // Delete structs
    delete h_cyl;
    delete d_cyl;

    delete h_cylPost;
    delete d_cylPost;

    sim.h_caseData = nullptr;
    sim.d_caseData = nullptr;
}

inline void copyHostToDevice(airfoilVar &d_airfoil, airfoilVar &h_airfoil)
{
    const int NB = h_airfoil.NB;
    const int NB_FLUID = h_airfoil.NB_FLUID;
    const int NB_SOLID = h_airfoil.NB_SOLID;

    d_airfoil.NB = NB;
    d_airfoil.NB_FLUID = NB_FLUID;
    d_airfoil.NB_SOLID = NB_SOLID;

    // Copy arrays (host → device)
    checkCudaErrors(cudaMemcpy(d_airfoil.boundaryList, h_airfoil.boundaryList, NB * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_airfoil.incomingMask, h_airfoil.incomingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_airfoil.outgoingMask, h_airfoil.outgoingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_airfoil.bcfluidList, h_airfoil.bcfluidList, NB_FLUID * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_airfoil.bcsolidList, h_airfoil.bcsolidList, NB_SOLID * sizeof(size_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_airfoil.unit_nx, h_airfoil.unit_nx, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_airfoil.unit_ny, h_airfoil.unit_ny, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_airfoil.delta_w, h_airfoil.delta_w, NB * sizeof(real), cudaMemcpyHostToDevice));
}

inline void airfoil_host_device_constants()
{
    cudaMemcpyToSymbol(d_incomingMask_bcfluid, h_incomingMask_bcfluid, MAX_NODE_TAG * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_outgoingMask_bcfluid, h_outgoingMask_bcfluid, MAX_NODE_TAG * sizeof(uint32_t));

    cudaMemcpyToSymbol(d_incomingMask_bcsolid, h_incomingMask_bcsolid, MAX_NODE_TAG * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_outgoingMask_bcsolid, h_outgoingMask_bcsolid, MAX_NODE_TAG * sizeof(uint32_t));
}

inline void write_forces_mass(const nodeVar &fMom, const int iter)
{
    cudaMemcpyFromSymbol(&h_TotalFx, d_TotalFx, sizeof(real));
    cudaMemcpyFromSymbol(&h_TotalFy, d_TotalFy, sizeof(real));
    cudaMemcpyFromSymbol(&h_TotalFz, d_TotalFz, sizeof(real));
    cudaMemcpyFromSymbol(&h_Totalm, d_Totalm, sizeof(real));

    write_forces(iter);
    write_mass_flux(iter);
}

inline void write_pressure(const airfoilVar &h_airfoil,
                           const airfoilPostProcess &h_airfoilPost)
{

    std::string filename = construct_path(PATH_FILES, ID_SIM, "pressure.dat");

    std::ofstream file(filename);
    file << std::setprecision(12);

    file << "theta z Cp\n";

    const int NB = h_airfoil.NB;

    for (int i = 0; i < NB; i++)
    {
        size_t idx = h_airfoil.boundaryList[i];

        unsigned int x, y, z;
        GlobalIndexToXYZ(idx, x, y, z);

        // compute theta
        real dx = x - XC;
        real dy = y - YC;

        real theta = atan2(dy, dx) * 180.0 / M_PI;
        if (theta < 0)
            theta += 360.0;

        real Cp = h_airfoilPost.Cp_avg[i];

        file<< std::setprecision(12) << theta << " " << z << " " << Cp << "\n";
    }

    file.close();
}