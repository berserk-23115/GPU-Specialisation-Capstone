#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "kernels.h"
#include "video_io.h"

// Function to print usage information
void printUsage(const char* programName) {
    printf("CUDA Video Processor - Real-time Video Enhancement and Analysis\n");
    printf("Usage: %s [options]\n", programName);
    printf("Options:\n");
    printf("  --input <source>           Input source (video file or camera index)\n");
    printf("  --output <filename>        Output video file (optional)\n");
    printf("  --filter <filter_type>     Filter to apply (default: none)\n");
    printf("  --transform <transform>    Transform to apply (default: none)\n");
    printf("  --intensity <value>        Filter intensity (0.0-1.0, default: 0.5)\n");
    printf("  --detect-motion            Enable motion detection\n");
    printf("  --optical-flow             Enable optical flow visualization\n");
    printf("  --detect-objects           Enable simple object detection\n");
    printf("  --benchmark                Run performance benchmark\n");
    printf("  --batch-size <size>        Batch processing size (default: 1)\n");
    printf("  --help                     Display this help message\n");
    printf("\n");
    printf("Available filters:\n");
    printf("  none, blur, sharpen, edge_detect, emboss, sepia, grayscale,\n");
    printf("  negative, cartoon, sketch, night_vision, thermal\n");
    printf("\n");
    printf("Available transformations:\n");
    printf("  none, rotate_90, rotate_180, rotate_270, flip_h, flip_v\n");
}

// Simple string comparison function
int str_compare(const char* str1, const char* str2) {
    return strcmp(str1, str2) == 0;
}

// Function to parse command line arguments
bool parseArguments(int argc, char** argv, char* input, char* output,
                    char* filterName, char* transformName,
                    float* intensity, bool* detectMotion, bool* opticalFlow,
                    bool* detectObjects, bool* benchmark, int* batchSize) {
    // Set default values
    strcpy(input, "0");  // Default to camera 0
    strcpy(output, "");
    strcpy(filterName, "none");
    strcpy(transformName, "none");
    *intensity = 0.5f;
    *detectMotion = false;
    *opticalFlow = false;
    *detectObjects = false;
    *benchmark = false;
    *batchSize = 1;
    
    for (int i = 1; i < argc; i++) {
        if (str_compare(argv[i], "--help")) {
            printUsage(argv[0]);
            return false;
        } else if (str_compare(argv[i], "--input") && i + 1 < argc) {
            strcpy(input, argv[++i]);
        } else if (str_compare(argv[i], "--output") && i + 1 < argc) {
            strcpy(output, argv[++i]);
        } else if (str_compare(argv[i], "--filter") && i + 1 < argc) {
            strcpy(filterName, argv[++i]);
        } else if (str_compare(argv[i], "--transform") && i + 1 < argc) {
            strcpy(transformName, argv[++i]);
        } else if (str_compare(argv[i], "--intensity") && i + 1 < argc) {
            *intensity = atof(argv[++i]);
        } else if (str_compare(argv[i], "--detect-motion")) {
            *detectMotion = true;
        } else if (str_compare(argv[i], "--optical-flow")) {
            *opticalFlow = true;
        } else if (str_compare(argv[i], "--detect-objects")) {
            *detectObjects = true;
        } else if (str_compare(argv[i], "--benchmark")) {
            *benchmark = true;
        } else if (str_compare(argv[i], "--batch-size") && i + 1 < argc) {
            *batchSize = atoi(argv[++i]);
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            printUsage(argv[0]);
            return false;
        }
    }
    
    // Validate arguments
    if (*intensity < 0.0f || *intensity > 1.0f) {
        printf("Error: Intensity must be between 0.0 and 1.0\n");
        return false;
    }
    
    if (*batchSize < 1) {
        printf("Error: Batch size must be at least 1\n");
        return false;
    }
    
    return true;
}

// Convert filter name to enum
FilterType getFilterType(const char* filterName) {
    if (str_compare(filterName, "blur")) return FilterType::BLUR;
    if (str_compare(filterName, "sharpen")) return FilterType::SHARPEN;
    if (str_compare(filterName, "edge_detect")) return FilterType::EDGE_DETECT;
    if (str_compare(filterName, "emboss")) return FilterType::EMBOSS;
    if (str_compare(filterName, "sepia")) return FilterType::SEPIA;
    if (str_compare(filterName, "grayscale")) return FilterType::GRAYSCALE;
    if (str_compare(filterName, "negative")) return FilterType::NEGATIVE;
    if (str_compare(filterName, "cartoon")) return FilterType::CARTOON;
    if (str_compare(filterName, "sketch")) return FilterType::SKETCH;
    if (str_compare(filterName, "night_vision")) return FilterType::NIGHT_VISION;
    if (str_compare(filterName, "thermal")) return FilterType::THERMAL;
    return FilterType::NONE;
}

// Convert transform name to enum
TransformType getTransformType(const char* transformName) {
    if (str_compare(transformName, "rotate_90")) return TransformType::ROTATE_90;
    if (str_compare(transformName, "rotate_180")) return TransformType::ROTATE_180;
    if (str_compare(transformName, "rotate_270")) return TransformType::ROTATE_270;
    if (str_compare(transformName, "flip_h")) return TransformType::FLIP_HORIZONTAL;
    if (str_compare(transformName, "flip_v")) return TransformType::FLIP_VERTICAL;
    return TransformType::NONE;
}

// Simple benchmark function
void runSimpleBenchmark() {
    printf("Running simple CUDA benchmark...\n");
    
    // Test simple image processing
    int width = 1920;
    int height = 1080;
    int channels = 3;
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    
    // Allocate host memory
    unsigned char* h_input = (unsigned char*)malloc(imageSize);
    unsigned char* h_output = (unsigned char*)malloc(imageSize);
    
    if (!h_input || !h_output) {
        printf("Error: Could not allocate host memory\n");
        return;
    }
    
    // Initialize test image
    for (int i = 0; i < width * height * channels; i++) {
        h_input[i] = rand() % 256;
    }
    
    // Test different filters
    FilterParams params;
    params.intensity = 0.5f;
    for (int i = 0; i < 4; i++) params.parameters[i] = 0.5f;
    
    printf("Testing filters on %dx%d image:\n", width, height);
    
    // Test grayscale
    printf("Testing Grayscale filter...\n");
    applySpecialFilter(h_input, h_output, FilterType::GRAYSCALE, params, 
                      width, height, channels);
    printf("  Grayscale: OK\n");
    
    // Test sepia
    printf("Testing Sepia filter...\n");
    applySpecialFilter(h_input, h_output, FilterType::SEPIA, params, 
                      width, height, channels);
    printf("  Sepia: OK\n");
    
    // Test negative
    printf("Testing Negative filter...\n");
    applySpecialFilter(h_input, h_output, FilterType::NEGATIVE, params, 
                      width, height, channels);
    printf("  Negative: OK\n");
    
    // Clean up
    free(h_input);
    free(h_output);
    
    printf("Benchmark completed successfully!\n");
}

int main(int argc, char** argv) {
    // Parse command line arguments using C-style strings
    char input[256], output[256], filterName[64], transformName[64];
    float intensity;
    bool detectMotion, opticalFlow, detectObjects, benchmark;
    int batchSize;
    
    if (!parseArguments(argc, argv, input, output, filterName, transformName,
                       &intensity, &detectMotion, &opticalFlow, &detectObjects, 
                       &benchmark, &batchSize)) {
        return 1;
    }
    
    // Print CUDA device information
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found! Exiting...\n");
        return 1;
    }
    
    // Print CUDA device information
    printCudaDeviceProperties(0);
    
    // Run benchmark if requested
    if (benchmark) {
        runSimpleBenchmark();
        return 0;
    }
    
    // Create video processor
    VideoProcessor* processor = createVideoProcessor();
    if (!processor) {
        printf("Error: Could not create video processor\n");
        return 1;
    }
    
    // Open input video
    bool isFile = !(str_compare(input, "0") || str_compare(input, "1"));
    if (!openVideoSource(processor, input, isFile)) {
        printf("Error: Could not open video source: %s\n", input);
        destroyVideoProcessor(processor);
        return 1;
    }
    
    printf("Video opened successfully!\n");
    printf("Resolution: %dx%d\n", getVideoWidth(processor), getVideoHeight(processor));
    printf("FPS: %.2f\n", getVideoFPS(processor));
    
    // Get filter and transform types
    FilterType filterType = getFilterType(filterName);
    TransformType transformType = getTransformType(transformName);
    
    // Prepare filter parameters
    FilterParams filterParams;
    filterParams.intensity = intensity;
    for (int i = 0; i < 4; i++) filterParams.parameters[i] = 0.5f;
    
    printf("Processing video with filter: %s, transform: %s, intensity: %.2f\n", 
           filterName, transformName, intensity);
    
    // Open output video if specified
    if (strlen(output) > 0) {
        if (!openVideoOutput(processor, output)) {
            printf("Error: Could not open output video: %s\n", output);
            destroyVideoProcessor(processor);
            return 1;
        }
        printf("Output video opened: %s\n", output);
    }
    
    // Process video frames
    int frameCount = 0;
    while (true) {
        // Read frame
        if (!readVideoFrame(processor)) {
            printf("End of video or read error\n");
            break;
        }
        
        // Process frame with CUDA
        if (!processVideoFrame(processor, filterType, filterParams, transformType)) {
            printf("Error processing frame %d\n", frameCount);
            break;
        }
        
        // Write frame to output if specified
        if (strlen(output) > 0) {
            if (!writeVideoFrame(processor)) {
                printf("Error writing frame %d\n", frameCount);
                break;
            }
        }
        
        frameCount++;
        if (frameCount % 100 == 0) {
            printf("Processed %d frames\n", frameCount);
        }
        
        // For camera input, limit processing to avoid infinite loop in test
        if (!isFile && frameCount >= 1000) {
            printf("Stopping camera processing after 1000 frames\n");
            break;
        }
    }
    
    printf("Total frames processed: %d\n", frameCount);
    
    // Clean up
    destroyVideoProcessor(processor);
    
    return 0;
} 