#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// Filter types
enum class FilterType {
    NONE,
    BLUR,
    SHARPEN,
    EDGE_DETECT,
    EMBOSS,
    SEPIA,
    GRAYSCALE,
    NEGATIVE,
    CARTOON,
    SKETCH,
    NIGHT_VISION,
    THERMAL
};

// Transformation types
enum class TransformType {
    NONE,
    ROTATE_90,
    ROTATE_180,
    ROTATE_270,
    FLIP_HORIZONTAL,
    FLIP_VERTICAL,
    PERSPECTIVE_WARP
};

// Structure for warp perspective parameters
typedef struct {
    float srcPoints[8]; // Source points (x1,y1,x2,y2,x3,y3,x4,y4)
    float dstPoints[8]; // Destination points (x1,y1,x2,y2,x3,y3,x4,y4)
} WarpPerspectiveParams;

// Structure for filter parameters
typedef struct {
    float intensity;    // General intensity parameter (0.0 - 1.0)
    float parameters[4]; // Additional parameters for specific filters
} FilterParams;

// Function to generate different types of filters
void generateFilter(float* filter, int filterWidth, FilterType filterType, const FilterParams& params);

// Host wrapper function for convolution kernel
void applyConvolution(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    const float* h_filter,
    int filterWidth,
    int imageWidth,
    int imageHeight,
    int imageChannels,
    cudaStream_t stream = 0
);

// Host wrapper for specialized filter kernels
void applySpecialFilter(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    FilterType filterType,
    const FilterParams& params,
    int imageWidth,
    int imageHeight,
    int imageChannels,
    cudaStream_t stream = 0
);

// Host wrapper for image transformation kernels
void applyTransformation(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    TransformType transformType,
    int imageWidth,
    int imageHeight,
    int imageChannels,
    const void* transformParams = nullptr,
    cudaStream_t stream = 0
);

// Batch process multiple frames (useful for video processing)
void batchProcessFrames(
    const unsigned char** h_inputFrames,
    unsigned char** h_outputFrames,
    int numFrames,
    int frameWidth,
    int frameHeight,
    int frameChannels,
    FilterType filterType,
    const FilterParams& params,
    TransformType transformType = TransformType::NONE,
    const void* transformParams = nullptr
);

// Motion detection between frames
void detectMotion(
    const unsigned char* previousFrame,
    const unsigned char* currentFrame,
    unsigned char* motionMask,
    int frameWidth,
    int frameHeight,
    float threshold,
    cudaStream_t stream = 0
);

// Optical flow computation between frames
void computeOpticalFlow(
    const unsigned char* previousFrame,
    const unsigned char* currentFrame,
    float* flowVectorsX,
    float* flowVectorsY,
    int frameWidth,
    int frameHeight,
    cudaStream_t stream = 0
);

// Smart object detection mask generation (simplified)
void generateObjectMask(
    const unsigned char* inputFrame,
    unsigned char* objectMask,
    int frameWidth,
    int frameHeight,
    float threshold,
    cudaStream_t stream = 0
);

#endif // KERNELS_H 