#pragma once

#include "cylinder_helpers.cuh"

void cylinder_initialize(Simulation &sim);
void cylinder_apply_boundary(Simulation &sim, int iter);
void cylinder_post_streaming(Simulation &sim, int iter);
void cylinder_post_collision(Simulation &sim, int iter);
void setup_cylinder_case(Case &case_module);

void cylinder_incoming_force_kernal(Simulation &sim, const int iter);
void cylinder_outgoing_force_kernal(Simulation &sim, const int iter);

void cylinder_post_process(Simulation &sim, int iter);

__device__ void cylinder_boundary_moments(nodeType_t nodeType, cylinderVar &cylinder, nodeVar &dMom, real *pop,
                                          real &rho, real &ux, real &uy, real &uz,
                                          real &mxx, real &myy, real &mzz,
                                          real &mxy, real &mxz, real &myz);

__global__ void apply_bc_cylinder(const int NB, const nodeType_t NODE_TYPE, const cylinderVar &cylinder,
                                  nodeVar dMom, const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                  const real D_WALL, const int iter);

__global__ void compute_surface_pressure(const nodeVar &dMom, const cylinderVar &d_cylinder,
                                         const cylinderPostProcess &d_cylinderPost,
                                         const int n_avg);

//=================================================================================================================

inline void allocateCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder,
                                   cylinderPostProcess &h_cylinderPost, cylinderPostProcess &d_cylinderPost)
{
    const int NB = h_cylinder.NB;
    const int NB_FLUID = h_cylinder.NB_FLUID;
    const int NB_SOLID = h_cylinder.NB_SOLID;

    if (NB <= 0)
    {
        std::cerr << "Error: NB not initialized!\n";
        exit(EXIT_FAILURE);
    }

    // ================= HOST =================
    checkCudaErrors(cudaMallocHost(&h_cylinder.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMallocHost(&h_cylinder.bcfluidList, NB_FLUID * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.bcsolidList, NB_SOLID * sizeof(size_t)));

    checkCudaErrors(cudaMallocHost(&h_cylinder.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_cylinder.delta_w, NB * sizeof(real)));

    checkCudaErrors(cudaMallocHost(&h_cylinderPost.Cp_avg, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_cylinderPost.Cp_rms_avg, NB * sizeof(real)));
    // checkCudaErrors(cudaMallocHost(&h_cylinderPost.Cp_span_avg, NB * sizeof(real)));

    memset(h_cylinderPost.Cp_avg, 0, NB * sizeof(real));
    memset(h_cylinderPost.Cp_rms_avg, 0, NB * sizeof(real));
    // memset(h_cylinderPost.Cp_span_avg, 0, NB * sizeof(real));
    h_cylinderPost.n_avg = 0;

    // ================= DEVICE =================
    checkCudaErrors(cudaMalloc(&d_cylinder.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMalloc(&d_cylinder.bcfluidList, NB_FLUID * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d_cylinder.bcsolidList, NB_SOLID * sizeof(size_t)));

    checkCudaErrors(cudaMalloc(&d_cylinder.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_cylinder.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_cylinder.delta_w, NB * sizeof(real)));

    checkCudaErrors(cudaMallocHost(&d_cylinderPost.Cp_avg, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&d_cylinderPost.Cp_rms_avg, NB * sizeof(real)));
    // checkCudaErrors(cudaMallocHost(&d_cylinderPost.Cp_span_avg, NB * sizeof(real)));

    cudaMemset(d_cylinderPost.Cp_avg, 0, NB * sizeof(real));
    cudaMemset(d_cylinderPost.Cp_rms_avg, 0, NB * sizeof(real));
    // cudaMemset(d_cylinderPost.Cp_span_avg, 0, NB * sizeof(real));
}

inline void freeCylinderMemory(cylinderVar &h_cylinder, cylinderVar &d_cylinder)
{
    // Free-ing host memory
    checkCudaErrors(cudaFreeHost(h_cylinder.boundaryList));
    checkCudaErrors(cudaFreeHost(h_cylinder.incomingMask));
    checkCudaErrors(cudaFreeHost(h_cylinder.outgoingMask));

    checkCudaErrors(cudaFreeHost(h_cylinder.bcfluidList));
    checkCudaErrors(cudaFreeHost(h_cylinder.bcsolidList));

    // Free-ing device memory
    checkCudaErrors(cudaFree(d_cylinder.boundaryList));
    checkCudaErrors(cudaFree(d_cylinder.incomingMask));
    checkCudaErrors(cudaFree(d_cylinder.outgoingMask));

    checkCudaErrors(cudaFree(d_cylinder.bcfluidList));
    checkCudaErrors(cudaFree(d_cylinder.bcsolidList));
}

inline void cylinder_free(Simulation &sim)
{
    auto *h_cyl = static_cast<cylinderVar *>(sim.h_caseData);
    auto *d_cyl = static_cast<cylinderVar *>(sim.d_caseData);
    auto *h_cylPost = static_cast<cylinderPostProcess *>(sim.h_casePost);
    auto *d_cylPost = static_cast<cylinderPostProcess *>(sim.d_casePost);

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

inline void copyHostToDevice(cylinderVar &d_cylinder, cylinderVar &h_cylinder)
{
    const int NB = h_cylinder.NB;
    const int NB_FLUID = h_cylinder.NB_FLUID;
    const int NB_SOLID = h_cylinder.NB_SOLID;

    d_cylinder.NB = NB;
    d_cylinder.NB_FLUID = NB_FLUID;
    d_cylinder.NB_SOLID = NB_SOLID;

    // Copy arrays (host → device)
    checkCudaErrors(cudaMemcpy(d_cylinder.boundaryList, h_cylinder.boundaryList, NB * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.incomingMask, h_cylinder.incomingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.outgoingMask, h_cylinder.outgoingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.bcfluidList, h_cylinder.bcfluidList, NB_FLUID * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.bcsolidList, h_cylinder.bcsolidList, NB_SOLID * sizeof(size_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_cylinder.unit_nx, h_cylinder.unit_nx, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.unit_ny, h_cylinder.unit_ny, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.delta_w, h_cylinder.delta_w, NB * sizeof(real), cudaMemcpyHostToDevice));
}

inline void cylinder_host_device_constants()
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

inline void write_pressure(const cylinderVar &h_cylinder,
                           const cylinderPostProcess &h_cylinderPost)
{

    std::string filename = construct_path(PATH_FILES, ID_SIM, "pressure.dat");

    std::ofstream file(filename);
    file << std::setprecision(12);

    file << "theta z Cp\n";

    const int NB = h_cylinder.NB;

    for (int i = 0; i < NB; i++)
    {
        size_t idx = h_cylinder.boundaryList[i];

        unsigned int x, y, z;
        GlobalIndexToXYZ(idx, x, y, z);

        // compute theta
        real dx = x - XC;
        real dy = y - YC;

        real theta = atan2(dy, dx) * 180.0 / M_PI;
        if (theta < 0)
            theta += 360.0;

        real Cp = h_cylinderPost.Cp_avg[i];

        file<< std::setprecision(12) << theta << " " << z << " " << Cp << "\n";
    }

    file.close();
}