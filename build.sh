#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting build process ---"

# Step 1: Detect the GPU compute capability using nvidia-smi
# The output is formatted as "X.Y" (e.g., 8.6)
GPU_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)

if [ -z "$GPU_CAPABILITY" ]; then
    echo "Error: Could not automatically detect GPU compute capability."
    echo "Please ensure you have NVIDIA drivers installed and nvidia-smi is in your PATH."
    exit 1
fi

# Format the capability for the nvcc -arch flag (e.g., from 8.6 to sm_86)
# We remove the dot and prepend 'sm_'
ARCH_FLAG="sm_${GPU_CAPABILITY//./}"

echo "Detected GPU Compute Capability: $GPU_CAPABILITY"
echo "Using nvcc architecture flag: -arch=$ARCH_FLAG"

# Step 2: Compile the CUDA code
# Command as requested, incorporating the dynamic architecture flag
nvcc -o rr.so \
     -O3 \
     --shared \
     -Xcompiler -fPIC \
     --use_fast_math \
     --compiler-options -fPIC \
     -arch=$ARCH_FLAG \
     rr.cu


nvcc -o liblooper.so \
     -O3 \
     --shared \
     -Xcompiler -fPIC \
     --use_fast_math \
     --compiler-options -fPIC \
     -arch=$ARCH_FLAG \
     liblooper.cu
echo "--- Compilation successful! ---"


