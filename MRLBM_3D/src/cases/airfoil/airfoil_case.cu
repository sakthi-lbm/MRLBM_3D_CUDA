#include "airfoil_kernels.cuh"

#ifdef AIRFOIL
__constant__ uint32_t d_incomingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcfluid[MAX_NODE_TAG];
__constant__ uint32_t d_incomingMask_bcsolid[MAX_NODE_TAG];
__constant__ uint32_t d_outgoingMask_bcsolid[MAX_NODE_TAG];

__device__ real d_TotalFx;
__device__ real d_TotalFy;
__device__ real d_TotalFz;
__device__ real d_Totalm;
#endif
void setup_airfoil_case(Case &case_module)
{
    case_module.initialize = airfoil_initialize;
    case_module.apply_boundary = airfoil_apply_boundary;

    case_module.post_streaming = airfoil_post_streaming;
    case_module.post_collision = airfoil_post_collision;
    case_module.post_process_step = airfoil_post_process;
    case_module.finalize = nullptr;

    case_module.free_case = airfoil_free;
}
