#include "annulus_kernels.cuh"

#ifdef ANNULUS
__constant__ uint32_t d_incomingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_incomingMask_bcsolid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcsolid[MAX_NODE_TAG];

__device__ real d_TotalFx;
__device__ real d_TotalFy;
__device__ real d_TotalFz;
__device__ real d_Totalm;

#endif

void setup_annulus_case(Case &case_module)
{
    case_module.initialize = annulus_initialize;
    case_module.apply_boundary = annulus_apply_boundary;

    case_module.post_streaming = nullptr;
    case_module.post_collision = nullptr;
    case_module.post_process_step = nullptr;
    case_module.finalize = nullptr;

    case_module.free_case = annulus_free;
}
