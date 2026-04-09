#pragma once

#include "cylinder_helpers.cuh"

#ifdef CYLINDER

void cylinder_initialize(Simulation &sim);
void cylinder_apply_boundary(Simulation &sim, int iter);
void cylinder_post_streaming(Simulation &sim, int iter);
void cylinder_post_collision(Simulation &sim, int iter);
void setup_cylinder_case(Case &case_module);

void cylinder_write_output(Simulation &sim, int iter);
void cylinder_incoming_force_kernal(Simulation &sim, const int iter);
void cylinder_outgoing_force_kernal(Simulation &sim, const int iter);

void cylinder_post_process(Simulation &sim, int iter);

__global__ void inlet_avg_density(const real *__restrict__ rho);
void compute_inlet_average_density(const real *d_rho);

__global__ void apply_bc_cylinder(const int NB, const boundaryVar &cylinder, nodeVar dMom,
                                  const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                  const real D_WALL, const int iter);

__global__ void cylinder_force_mass_kernel(const nodeVar dMom,
                                           const cylinderVar *cylinder,
                                           const int NB, const int NB_FLUID,
                                           const MaskType masktype,
                                           const real sign);

__global__ void cylinder_surface_pressure_kernel(const nodeVar dMom,
                                                 const cylinderVar *cylinder,
                                                 cylinderPostProcess *cylinderPost,
                                                 const int NB,
                                                 const int n_avg);

__global__ void compute_surface_pressure(const nodeVar &dMom, const int nb,
                                         size_t *boundaryList, real *d_unit_nx, real *d_unit_ny,
                                         real *d_ps_avg, const int n_avg);

//=================================================================================================================

__device__ __forceinline__ void cylinder_boundary_moments(const nodeType_t nodeType_packed,
                                                          const boundaryVar &inner,
                                                          nodeVar &dMom, real *pop, real *s_pop,
                                                          real &rho, real &ux, real &uy, real &uz,
                                                          real &mxx, real &myy, real &mzz,
                                                          real &mxy, real &mxz, real &myz)
{
    const nodeType_t nodeType = getType(nodeType_packed);
    const nodeType_t tag = getIndex(nodeType_packed);

    if (nodeType == NODE_INNER)
    {
        const real unit_nx = inner.unit_nx[tag];
        const real unit_ny = inner.unit_ny[tag];
        const uint32_t incomingMask = inner.incomingMask[tag];

        evaluate_incoming_moments_rotated(unit_nx, unit_ny, incomingMask, pop, rho, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (triangular && (nodeType == NODE_BCFLUID_INNER))
    {
        // printf("%d %d \n", toInt(nodeType), toInt(tag));
        const uint32_t incomingMask = d_incomingMask_bcfluid[tag];

        fluid_boundary_condition(tag, incomingMask, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (triangular && !Z_PERIODIC && (nodeType == NODE_BCSOLID_INNER))
    {
        bcsolid_boundary_condition(tag, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        boundary_condition(nodeType, dMom, pop, s_pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
}

inline void allocatecylinderMemory_Host(boundaryVar &h)
{
    int NB = h.NB;
    if (NB <= 0)
    {
        std::cerr << "Error: NB not initialized!\n";
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMallocHost(&h.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMallocHost(&h.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMallocHost(&h.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMallocHost(&h.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h.delta_w, NB * sizeof(real)));

    if constexpr (triangular)
    {
        const int NB_FLUID = h.NB_FLUID;
        const int NB_SOLID = h.NB_SOLID;

        checkCudaErrors(cudaMallocHost(&h.bcfluidList, NB_FLUID * sizeof(size_t)));

        if (!Z_PERIODIC)
        {
            checkCudaErrors(cudaMallocHost(&h.bcsolidList, NB_SOLID * sizeof(size_t)));
        }
    }
}

inline void allocatecylinderMemory_Device(boundaryVar &d, const boundaryVar &h)
{
    int NB = h.NB;

    d.NB = h.NB;

    checkCudaErrors(cudaMalloc(&d.boundaryList, NB * sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&d.incomingMask, NB * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&d.outgoingMask, NB * sizeof(uint32_t)));

    checkCudaErrors(cudaMalloc(&d.unit_nx, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d.unit_ny, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d.delta_w, NB * sizeof(real)));

    if constexpr (triangular)
    {
        const int NB_FLUID = h.NB_FLUID;
        const int NB_SOLID = h.NB_SOLID;

        d.NB_FLUID = h.NB_FLUID;
        d.NB_SOLID = h.NB_SOLID;

        checkCudaErrors(cudaMalloc(&d.bcfluidList, NB_FLUID * sizeof(size_t)));

        if (!Z_PERIODIC)
        {
            checkCudaErrors(cudaMalloc(&d.bcsolidList, NB_SOLID * sizeof(size_t)));
        }
    }
}

inline void copyCylinder_HostToDevice(boundaryVar &d, const boundaryVar &h)
{
    int NB = h.NB;

    cudaMemcpy(d.boundaryList, h.boundaryList, NB * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d.incomingMask, h.incomingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d.outgoingMask, h.outgoingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d.unit_nx, h.unit_nx, NB * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d.unit_ny, h.unit_ny, NB * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d.delta_w, h.delta_w, NB * sizeof(real), cudaMemcpyHostToDevice);

    if constexpr (triangular)
    {
        const int NB_FLUID = h.NB_FLUID;
        const int NB_SOLID = h.NB_SOLID;

        cudaMemcpy(d.bcfluidList, h.bcfluidList, NB_FLUID * sizeof(size_t), cudaMemcpyHostToDevice);

        if (!Z_PERIODIC)
        {
            cudaMemcpy(d.bcsolidList, h.bcsolidList, NB_SOLID * sizeof(size_t), cudaMemcpyHostToDevice);
        }
    }
}

inline void allocateCylinderPost_Host(cylinderPostProcess &h_post, int NB)
{
    if (NB <= 0)
    {
        std::cerr << "Error: NB not initialized for post-process!\n";
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMallocHost(&h_post.ps_avg, NB * sizeof(real)));
    checkCudaErrors(cudaMallocHost(&h_post.ps_rms_avg, NB * sizeof(real)));

    memset(h_post.ps_avg, 0, NB * sizeof(real));
    memset(h_post.ps_rms_avg, 0, NB * sizeof(real));

    h_post.n_avg = 0;
}

inline void allocateCylinderPost_Device(cylinderPostProcess &d_post, int NB)
{
    if (NB <= 0)
    {
        std::cerr << "Error: NB not initialized for device post-process!\n";
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(&d_post.ps_avg, NB * sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_post.ps_rms_avg, NB * sizeof(real)));

    cudaMemset(d_post.ps_avg, 0, NB * sizeof(real));
    cudaMemset(d_post.ps_rms_avg, 0, NB * sizeof(real));

    d_post.n_avg = 0;
}

inline void copyCylinderPost_HostToDevice(cylinderPostProcess &d_post, const cylinderPostProcess &h_post, int NB)
{
    cudaMemcpy(d_post.ps_avg, h_post.ps_avg, NB * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_post.ps_rms_avg, h_post.ps_rms_avg, NB * sizeof(real), cudaMemcpyHostToDevice);
    d_post.n_avg = h_post.n_avg; // scalar copy
}

inline void copyCylinderPost_DeviceToHost(cylinderPostProcess &h_post, const cylinderPostProcess &d_post, int NB)
{
    cudaMemcpy(h_post.ps_avg, d_post.ps_avg, NB * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_post.ps_rms_avg, d_post.ps_rms_avg, NB * sizeof(real), cudaMemcpyDeviceToHost);
    h_post.n_avg = d_post.n_avg;
}

inline void free_boundary(boundaryVar &h_b, boundaryVar &d_b)
{
    // host
    cudaFreeHost(h_b.boundaryList);
    cudaFreeHost(h_b.incomingMask);
    cudaFreeHost(h_b.outgoingMask);

    if (h_b.bcfluidList)
        cudaFreeHost(h_b.bcfluidList);
    if (h_b.bcsolidList)
        cudaFreeHost(h_b.bcsolidList);

    cudaFreeHost(h_b.unit_nx);
    cudaFreeHost(h_b.unit_ny);
    cudaFreeHost(h_b.delta_w);

    // device
    cudaFree(d_b.boundaryList);
    cudaFree(d_b.incomingMask);
    cudaFree(d_b.outgoingMask);

    if (d_b.bcfluidList)
        cudaFree(d_b.bcfluidList);
    if (d_b.bcsolidList)
        cudaFree(d_b.bcsolidList);

    cudaFree(d_b.unit_nx);
    cudaFree(d_b.unit_ny);
    cudaFree(d_b.delta_w);
}

inline void free_boundaryPost(cylinderPostProcess &h_Post, cylinderPostProcess &d_Post)
{
    // host
    cudaFreeHost(h_Post.ps_avg);
    cudaFreeHost(h_Post.ps_rms_avg);

    // device
    cudaFree(d_Post.ps_avg);
    cudaFree(d_Post.ps_rms_avg);
}

inline void cylinder_free(Simulation &sim)
{

    auto *h_cylinder = static_cast<cylinderVar *>(sim.h_caseData);
    auto *d_cylinder = static_cast<cylinderVar *>(sim.d_caseData);

    auto *h_cylinderPost = static_cast<cylinderPostProcess *>(sim.h_casePost);
    auto *d_cylinderPost = static_cast<cylinderPostProcess *>(sim.d_casePost);

    if (h_cylinder && d_cylinder)
    {
        free_boundary(h_cylinder->inner, d_cylinder->inner);

        delete h_cylinder;
        delete d_cylinder;

        sim.h_caseData = nullptr;
        sim.d_caseData = nullptr;
    }

    // -------- post-process --------
    if (h_cylinderPost && d_cylinderPost)
    {
        free_boundaryPost(*h_cylinderPost, *d_cylinderPost);

        delete h_cylinderPost;
        delete d_cylinderPost;

        sim.h_casePost = nullptr;
        sim.d_casePost = nullptr;
    }
}

inline void copyHostToDevice(boundaryVar &d_cylinder, boundaryVar &h_cylinder)
{
    const int NB = h_cylinder.NB;
    d_cylinder.NB = NB;

    // Copy arrays (host → device)
    checkCudaErrors(cudaMemcpy(d_cylinder.boundaryList, h_cylinder.boundaryList, NB * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.incomingMask, h_cylinder.incomingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.outgoingMask, h_cylinder.outgoingMask, NB * sizeof(uint32_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_cylinder.unit_nx, h_cylinder.unit_nx, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.unit_ny, h_cylinder.unit_ny, NB * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cylinder.delta_w, h_cylinder.delta_w, NB * sizeof(real), cudaMemcpyHostToDevice));

    if (triangular)
    {
        const int NB_FLUID = h_cylinder.NB_FLUID;
        const int NB_SOLID = h_cylinder.NB_SOLID;
        d_cylinder.NB_FLUID = NB_FLUID;
        d_cylinder.NB_SOLID = NB_SOLID;

        checkCudaErrors(cudaMemcpy(d_cylinder.bcfluidList, h_cylinder.bcfluidList, NB_FLUID * sizeof(size_t), cudaMemcpyHostToDevice));
        if (!Z_PERIODIC)
            checkCudaErrors(cudaMemcpy(d_cylinder.bcsolidList, h_cylinder.bcsolidList, NB_SOLID * sizeof(size_t), cudaMemcpyHostToDevice));
    }
}

inline void cylinder_host_device_constants()
{
    cudaMemcpyToSymbol(d_incomingMask_bcfluid, h_incomingMask_bcfluid, MAX_NODE_TAG * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_outgoingMask_bcfluid, h_outgoingMask_bcfluid, MAX_NODE_TAG * sizeof(uint32_t));

    if (!Z_PERIODIC)
    {
        cudaMemcpyToSymbol(d_incomingMask_bcsolid, h_incomingMask_bcsolid, MAX_NODE_TAG * sizeof(uint32_t));
        cudaMemcpyToSymbol(d_outgoingMask_bcsolid, h_outgoingMask_bcsolid, MAX_NODE_TAG * sizeof(uint32_t));
    }
}

//================================ POST-PROCESS ====================================================
inline void write_forces_mass_density(const nodeVar &fMom, const int iter)
{
    cudaMemcpyFromSymbol(&h_TotalFx, d_TotalFx, sizeof(real));
    cudaMemcpyFromSymbol(&h_TotalFy, d_TotalFy, sizeof(real));
    cudaMemcpyFromSymbol(&h_TotalFz, d_TotalFz, sizeof(real));
    cudaMemcpyFromSymbol(&h_Totalm, d_Totalm, sizeof(real));

    write_forces(iter);
    write_mass_flux(iter);
    write_average_inlet_density(iter);
}

inline void write_pressure(const cylinderVar &h_cylinder,
                           const cylinderPostProcess &h_cylinderPost)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "pressure.dat");

    std::ofstream file(filename);
    file << std::setprecision(12);

    file << "theta z Cp\n";

    const int NB = h_cylinder.inner.NB;

    for (int i = 0; i < NB; i++)
    {
        size_t idx = h_cylinder.inner.boundaryList[i];

        unsigned int x, y, z;
        GlobalIndexToXYZ(idx, x, y, z);

        // compute theta
        real dx = x - XC;
        real dy = y - YC;

        real theta = atan2(dy, dx) * 180.0 / PI;
        if (theta < 0)
            theta += 360.0;

        real ps = h_cylinderPost.ps_avg[i];

        file << std::setprecision(12) << theta << " " << z << " " << ps << "\n";
    }

    file.close();
}

#endif
