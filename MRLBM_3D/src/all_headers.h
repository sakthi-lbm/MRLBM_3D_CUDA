    #ifndef ALL_HEADERS_H
    #define ALL_HEADERS_H

    #include "config.h"

    // clang-format off

    #define LATTICE_PROPERTIES "solver/latticeProperties.cuh"
    #include LATTICE_PROPERTIES
    #include CASE_CONSTANTS
    #include CASE_OUTPUTS
    #include RECONSTRUCT

    // clang-format on

    constexpr size_t MAX_THREADS_PER_BLOCK = 1024;
    constexpr dim3 findOptimalBlockDim(size_t maxShareMemBytes, size_t bytesPerThread)
    {
        size_t bestDimX = 1;
        size_t bestDimY = 1;
        size_t maxThreadsUsed = 0;
        for (size_t dimY = 2; dimY <= 32; dimY *= 2)
        {
            for (size_t dimX = 2; dimX <= 32; dimX *= 2)
            {
                size_t threads = dimX * dimY;
                size_t usedMem = threads * bytesPerThread;

                if (threads > MAX_THREADS_PER_BLOCK)
                    continue;

                if (usedMem <= maxShareMemBytes)
                {
                    if (threads > maxThreadsUsed)
                    {
                        bestDimX = dimX;
                        bestDimY = dimY;
                        maxThreadsUsed = threads;
                    }
                }
            }
        }
        return dim3(bestDimX, bestDimY);
    }

    #include "definitions.h"
    #include "index.h"
    #include "globalStructs.h"
    #include "nodeTypeMap.h"
    #include "utils/cudaHelpers.cuh"
    #include "utils/file_utils.h"

    #include CASE_BOUNDARY

    #endif // ALL_HEADERS_H