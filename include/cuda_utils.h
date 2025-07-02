#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA error checking function
void checkCudaError(cudaError_t code, const char *file, int line);

// Macro for CUDA error checking
#define CUDA_CHECK_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }

// Simple structure for device properties
typedef struct {
    char name[256];
    int major;
    int minor;
    int multiProcessorCount;
    size_t totalGlobalMem;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    int warpSize;
} CudaDeviceProps;

// Get CUDA device properties
void getCudaDeviceProperties(CudaDeviceProps* deviceProps, int deviceId);

// Print CUDA device properties
void printCudaDeviceProperties(int deviceId);

// Calculate optimal grid and block dimensions for a 2D problem
void calculateOptimalDimensions(
    int width, 
    int height, 
    dim3* blockDim, 
    dim3* gridDim,
    int preferredBlockSize
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_UTILS_H 