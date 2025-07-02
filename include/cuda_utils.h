#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " 
                  << file << ":" << line << std::endl;
        exit(code);
    }
}

// Device properties structure
struct CudaDeviceProps {
    std::string name;
    int major;
    int minor;
    int multiProcessorCount;
    size_t totalGlobalMem;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    int warpSize;
};

// Get CUDA device properties
inline CudaDeviceProps getCudaDeviceProperties(int deviceId = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    
    CudaDeviceProps deviceProps;
    deviceProps.name = prop.name;
    deviceProps.major = prop.major;
    deviceProps.minor = prop.minor;
    deviceProps.multiProcessorCount = prop.multiProcessorCount;
    deviceProps.totalGlobalMem = prop.totalGlobalMem;
    deviceProps.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    deviceProps.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    deviceProps.warpSize = prop.warpSize;
    
    return deviceProps;
}

// Print CUDA device properties
inline void printCudaDeviceProperties(int deviceId = 0) {
    CudaDeviceProps props = getCudaDeviceProperties(deviceId);
    
    std::cout << "CUDA Device Properties:" << std::endl;
    std::cout << "  Name: " << props.name << std::endl;
    std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "  Number of SMs: " << props.multiProcessorCount << std::endl;
    std::cout << "  Total Global Memory: " << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Max Threads Per Block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads Per SM: " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Warp Size: " << props.warpSize << std::endl;
}

// Calculate optimal grid and block dimensions for a 2D problem
inline void calculateOptimalDimensions(
    int width, 
    int height, 
    dim3 &blockDim, 
    dim3 &gridDim,
    int preferredBlockSize = 16) 
{
    blockDim.x = preferredBlockSize;
    blockDim.y = preferredBlockSize;
    blockDim.z = 1;
    
    gridDim.x = (width + blockDim.x - 1) / blockDim.x;
    gridDim.y = (height + blockDim.y - 1) / blockDim.y;
    gridDim.z = 1;
}

// Benchmark CUDA kernel execution time
template<typename Func>
inline float benchmarkKernel(Func kernelFunction) {
    cudaEvent_t start, stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    
    CUDA_CHECK_ERROR(cudaEventRecord(start));
    kernelFunction();
    CUDA_CHECK_ERROR(cudaEventRecord(stop));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CUDA_CHECK_ERROR(cudaEventDestroy(start));
    CUDA_CHECK_ERROR(cudaEventDestroy(stop));
    
    return milliseconds;
}

#endif // CUDA_UTILS_H 