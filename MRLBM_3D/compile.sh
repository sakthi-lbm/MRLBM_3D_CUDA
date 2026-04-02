#!/bin/bash

LT="D2Q9"
    # use 'ldc' as default if not given

if [ $# -lt 1 ]; then
    echo "❌ Error: No simulation ID provided."
    echo "Usage: bash compile.sh <SIM_ID>"
    exit 1
fi
ID_SIM="$1"



MODE=${2:-release}

if [ "$MODE" = "debug" ]; then
    echo "🔧 Debug build"
    FLAGS="-g -G -lineinfo -O0"
elif [ "$MODE" = "sanitize" ]; then
    echo "🧪 Sanitizer build"
    FLAGS="-g -lineinfo -O1"
else
    echo "Release build"
    FLAGS="-O3"
fi

# Detect GPU compute capability if not manually set
if [ -z "$CompCap" ]; then
    CompCap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -z "$CompCap" ]; then
        echo "Error: Unable to determine compute capability."
        exit 1
    fi
fi

echo "Building for Compute Capability sm_${CompCap}, ID_SIM: ${ID_SIM}, Lattice: ${LT}"

# Cleanup old binaries and outputs safely
rm -f ../*sim_${LT}_sm${CompCap} 2>/dev/null

# Compile all .cu files inside src/
nvcc -std=c++17 \
    -gencode arch=compute_${CompCap},code=sm_${CompCap} \
    -rdc=true $FLAGS --restrict \
    -Xptxas -v \
     -I. \
    -Iinclude \
    -Isrc \
    -Isolver \
    -D ID_SIM=\"$ID_SIM\" \
    $(find src -name "*.cu") \
    -lcudadevrt -lcurand \
    -o ../${ID_SIM}sim_${LT}_sm${CompCap}


# -Xptxas -v \
# Run the simulation
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="log_run_${ID_SIM}_${MODE}.log"
# LOGFILE="log_run_${ID_SIM}_${MODE}_${TIMESTAMP}.log"

cd ../
if [ "$MODE" = "sanitize" ]; then
    echo "Running with CUDA sanitizer"
    compute-sanitizer --tool memcheck --show-backtrace ./${ID_SIM}sim_${LT}_sm${CompCap} 2>&1 | tee ${LOGFILE}
else
    ./${ID_SIM}sim_${LT}_sm${CompCap} 2>&1 | tee ${LOGFILE}
fi
