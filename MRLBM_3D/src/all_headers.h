#ifndef ALL_HEADERS_H
#define ALL_HEADERS_H

#include "config.h"

// clang-format off

    #include CASE_CONSTANTS
    #include CASE_OUTPUTS
    #include RECONSTRUCT

// clang-format on

#include "definitions.h"
#include "index.h"
#include "globalStructs.h"
#include "nodeTypeMap.h"
#include "utils/cudaHelpers.cuh"
#include "utils/file_utils.h"

// #include CASE_BOUNDARY

#endif // ALL_HEADERS_H