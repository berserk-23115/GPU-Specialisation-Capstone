#include "cuda_utils.h"
#include <cstdio>
#include <cstring>

// CUDA error checking function implementation
void checkCudaError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Get CUDA device properties implementation
void getCudaDeviceProperties(CudaDeviceProps* deviceProps, int deviceId) {
    if (!deviceProps) return;
    
    cudaDeviceProp prop;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    
    strncpy(deviceProps->name, prop.name, 255);
    deviceProps->name[255] = '\0';  // Ensure null termination
    deviceProps->major = prop.major;
    deviceProps->minor = prop.minor;
    deviceProps->multiProcessorCount = prop.multiProcessorCount;
    deviceProps->totalGlobalMem = prop.totalGlobalMem;
    deviceProps->maxThreadsPerBlock = prop.maxThreadsPerBlock;
    deviceProps->maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    deviceProps->warpSize = prop.warpSize;
}

// Print CUDA device properties implementation
void printCudaDeviceProperties(int deviceId) {
    CudaDeviceProps props;
    getCudaDeviceProperties(&props, deviceId);
    
    printf("CUDA Device Properties:\n");
    printf("  Name: %s\n", props.name);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  Number of SMs: %d\n", props.multiProcessorCount);
    printf("  Total Global Memory: %zu MB\n", props.totalGlobalMem / (1024 * 1024));
    printf("  Max Threads Per Block: %d\n", props.maxThreadsPerBlock);
    printf("  Max Threads Per SM: %d\n", props.maxThreadsPerMultiProcessor);
    printf("  Warp Size: %d\n", props.warpSize);
}

// Calculate optimal grid and block dimensions implementation
void calculateOptimalDimensions(
    int width, 
    int height, 
    dim3* blockDim, 
    dim3* gridDim,
    int preferredBlockSize
) {
    if (!blockDim || !gridDim) return;
    
    blockDim->x = preferredBlockSize;
    blockDim->y = preferredBlockSize;
    blockDim->z = 1;
    
    gridDim->x = (width + blockDim->x - 1) / blockDim->x;
    gridDim->y = (height + blockDim->y - 1) / blockDim->y;
    gridDim->z = 1;
} 