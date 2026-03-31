#include "cylinder_kernels.cuh"

#ifdef CYLINDER
__constant__ uint32_t d_incomingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_incomingMask_bcsolid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcsolid[MAX_NODE_TAG];

__device__ real d_TotalFx;
__device__ real d_TotalFy;
__device__ real d_TotalFz;
__device__ real d_Totalm;

void setup_cylinder_case(Case &case_module)
{
    case_module.initialize = cylinder_initialize;
    case_module.apply_boundary = cylinder_apply_boundary;

    case_module.post_streaming = nullptr;
    case_module.post_collision = nullptr;
    case_module.post_process_step = nullptr;
    case_module.write_output = cylinder_write_output;
    case_module.finalize = nullptr;

    case_module.free_case = cylinder_free;
}

#endif
