#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_utils.h"
#include "kernels.h"
#include "video_io.h"

// Function to print usage information
void printUsage(const char* programName) {
    std::cout << "CUDA Video Processor - Real-time Video Enhancement and Analysis" << std::endl;
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input <source>           Input source (video file or camera index)" << std::endl;
    std::cout << "  --output <filename>        Output video file (optional)" << std::endl;
    std::cout << "  --filter <filter_type>     Filter to apply (default: none)" << std::endl;
    std::cout << "  --transform <transform>    Transform to apply (default: none)" << std::endl;
    std::cout << "  --intensity <value>        Filter intensity (0.0-1.0, default: 0.5)" << std::endl;
    std::cout << "  --detect-motion            Enable motion detection" << std::endl;
    std::cout << "  --optical-flow             Enable optical flow visualization" << std::endl;
    std::cout << "  --detect-objects           Enable simple object detection" << std::endl;
    std::cout << "  --benchmark                Run performance benchmark" << std::endl;
    std::cout << "  --batch-size <size>        Batch processing size (default: 1)" << std::endl;
    std::cout << "  --help                     Display this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Available filters:" << std::endl;
    std::cout << "  none, blur, sharpen, edge_detect, emboss, sepia, grayscale, " << std::endl;
    std::cout << "  negative, cartoon, sketch, night_vision, thermal" << std::endl;
    std::cout << std::endl;
    std::cout << "Available transformations:" << std::endl;
    std::cout << "  none, rotate_90, rotate_180, rotate_270, flip_h, flip_v" << std::endl;
}

// Function to parse command line arguments
bool parseArguments(int argc, char** argv, std::string& input, std::string& output,
                    std::string& filterName, std::string& transformName,
                    float& intensity, bool& detectMotion, bool& opticalFlow,
                    bool& detectObjects, bool& benchmark, int& batchSize) {
    // Set default values
    input = "0";  // Default to camera 0
    output = "";
    filterName = "none";
    transformName = "none";
    intensity = 0.5f;
    detectMotion = false;
    opticalFlow = false;
    detectObjects = false;
    benchmark = false;
    batchSize = 1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return false;
        } else if (arg == "--input" && i + 1 < argc) {
            input = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            filterName = argv[++i];
        } else if (arg == "--transform" && i + 1 < argc) {
            transformName = argv[++i];
        } else if (arg == "--intensity" && i + 1 < argc) {
            intensity = std::stof(argv[++i]);
        } else if (arg == "--detect-motion") {
            detectMotion = true;
        } else if (arg == "--optical-flow") {
            opticalFlow = true;
        } else if (arg == "--detect-objects") {
            detectObjects = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batchSize = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    
    // Validate arguments
    if (intensity < 0.0f || intensity > 1.0f) {
        std::cerr << "Error: Intensity must be between 0.0 and 1.0" << std::endl;
        return false;
    }
    
    if (batchSize < 1) {
        std::cerr << "Error: Batch size must be at least 1" << std::endl;
        return false;
    }
    
    return true;
}

// Convert filter name to enum
FilterType getFilterType(const std::string& filterName) {
    if (filterName == "blur") return FilterType::BLUR;
    if (filterName == "sharpen") return FilterType::SHARPEN;
    if (filterName == "edge_detect") return FilterType::EDGE_DETECT;
    if (filterName == "emboss") return FilterType::EMBOSS;
    if (filterName == "sepia") return FilterType::SEPIA;
    if (filterName == "grayscale") return FilterType::GRAYSCALE;
    if (filterName == "negative") return FilterType::NEGATIVE;
    if (filterName == "cartoon") return FilterType::CARTOON;
    if (filterName == "sketch") return FilterType::SKETCH;
    if (filterName == "night_vision") return FilterType::NIGHT_VISION;
    if (filterName == "thermal") return FilterType::THERMAL;
    return FilterType::NONE;
}

// Convert transform name to enum
TransformType getTransformType(const std::string& transformName) {
    if (transformName == "rotate_90") return TransformType::ROTATE_90;
    if (transformName == "rotate_180") return TransformType::ROTATE_180;
    if (transformName == "rotate_270") return TransformType::ROTATE_270;
    if (transformName == "flip_h") return TransformType::FLIP_HORIZONTAL;
    if (transformName == "flip_v") return TransformType::FLIP_VERTICAL;
    return TransformType::NONE;
}

// Function to run benchmarks
void runBenchmark(VideoProcessor& processor, const std::string& input) {
    std::cout << "Running performance benchmark..." << std::endl;
    
    // Open test video
    bool isFile = (input != "0" && input != "1");
    if (!processor.openVideo(input, isFile)) {
        std::cerr << "Error: Could not open video source for benchmarking" << std::endl;
        return;
    }
    
    cv::Mat frame, outputFrame;
    if (!processor.readFrame(frame)) {
        std::cerr << "Error: Could not read frame for benchmarking" << std::endl;
        return;
    }
    
    // Benchmark different filters
    std::vector<FilterType> filters = {
        FilterType::BLUR, FilterType::SHARPEN, FilterType::EDGE_DETECT,
        FilterType::EMBOSS, FilterType::SEPIA, FilterType::GRAYSCALE,
        FilterType::NEGATIVE, FilterType::CARTOON, FilterType::SKETCH
    };
    
    std::vector<std::string> filterNames = {
        "Blur", "Sharpen", "Edge Detection",
        "Emboss", "Sepia", "Grayscale",
        "Negative", "Cartoon", "Sketch"
    };
    
    // Prepare filter parameters
    FilterParams params;
    params.intensity = 0.5f;
    for (int i = 0; i < 4; i++) params.parameters[i] = 0.5f;
    
    std::cout << "Filter Benchmarks (1000x1 frame):" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    for (size_t i = 0; i < filters.size(); i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process frame 1000 times to get a reliable measurement
        for (int j = 0; j < 1000; j++) {
            processor.processFrame(frame, outputFrame, filters[i], params);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << filterNames[i] << ": " << duration << " ms (for 1000 frames)" << std::endl;
        std::cout << "   Average per frame: " << duration / 1000.0 << " ms" << std::endl;
        std::cout << "   FPS equivalent: " << 1000.0 / (duration / 1000.0) << std::endl;
    }
    
    // Test batch processing
    std::vector<cv::Mat> batchFrames(5, frame);
    std::vector<cv::Mat> batchOutputs(5);
    
    std::cout << "\nBatch Processing Benchmarks (5-frame batch):" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    
    for (size_t i = 0; i < filters.size(); i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process batch 200 times
        for (int j = 0; j < 200; j++) {
            processor.processBatch(batchFrames, batchOutputs, filters[i], params);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << filterNames[i] << ": " << duration << " ms (for 200*5=1000 frames)" << std::endl;
        std::cout << "   Average per frame: " << duration / 1000.0 << " ms" << std::endl;
        std::cout << "   FPS equivalent: " << 1000.0 / (duration / 1000.0) << std::endl;
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string input, output, filterName, transformName;
    float intensity;
    bool detectMotion, opticalFlow, detectObjects, benchmark;
    int batchSize;
    
    if (!parseArguments(argc, argv, input, output, filterName, transformName,
                       intensity, detectMotion, opticalFlow, detectObjects, 
                       benchmark, batchSize)) {
        return 1;
    }
    
    // Print CUDA device information
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found! Exiting..." << std::endl;
        return 1;
    }
    
    // Print CUDA device information
    printCudaDeviceProperties(0);
    
    // Create video processor
    VideoProcessor processor;
    
    // Run benchmark if requested
    if (benchmark) {
        runBenchmark(processor, input);
        return 0;
    }
    
    // Open input video
    bool isFile = (input != "0" && input != "1");  // Check if input is a file or camera
    if (!processor.openVideo(input, isFile)) {
        std::cerr << "Error: Could not open video source: " << input << std::endl;
        return 1;
    }
    
    std::cout << "Video source opened successfully:" << std::endl;
    std::cout << "  Width: " << processor.getWidth() << std::endl;
    std::cout << "  Height: " << processor.getHeight() << std::endl;
    std::cout << "  FPS: " << processor.getFPS() << std::endl;
    if (isFile) {
        std::cout << "  Total frames: " << processor.getTotalFrames() << std::endl;
    }
    
    // Open output video if specified
    if (!output.empty() && !processor.openOutputVideo(output)) {
        std::cerr << "Error: Could not open output video file: " << output << std::endl;
        return 1;
    }
    
    // Convert filter and transform names to enums
    FilterType filterType = getFilterType(filterName);
    TransformType transformType = getTransformType(transformName);
    
    // Prepare filter parameters
    FilterParams filterParams;
    filterParams.intensity = intensity;
    for (int i = 0; i < 4; i++) filterParams.parameters[i] = 0.5f;
    
    std::cout << "Processing with:" << std::endl;
    std::cout << "  Filter: " << filterName << std::endl;
    std::cout << "  Transform: " << transformName << std::endl;
    std::cout << "  Intensity: " << intensity << std::endl;
    
    // Batch processing settings
    std::vector<cv::Mat> batchFrames;
    std::vector<cv::Mat> batchOutputs(batchSize);
    
    // Main processing loop
    cv::Mat frame, outputFrame;
    cv::Mat motionMask, flowImage, objectMask;
    
    bool isRunning = true;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (isRunning) {
        // Read frame
        if (!processor.readFrame(frame)) {
            break;
        }
        
        // Create output frame
        outputFrame = frame.clone();
        
        // Process frame
        if (batchSize == 1) {
            // Single frame processing
            processor.processFrame(frame, outputFrame, filterType, filterParams, transformType);
        } else {
            // Batch processing
            batchFrames.push_back(frame.clone());
            
            if ((int)batchFrames.size() == batchSize) {
                // Process batch
                processor.processBatch(batchFrames, batchOutputs, filterType, filterParams, transformType);
                
                // Write processed frames
                for (const auto& outputFrame : batchOutputs) {
                    if (!output.empty()) {
                        processor.writeFrame(outputFrame);
                    }
                }
                
                // Display only the last frame
                outputFrame = batchOutputs.back().clone();
                
                // Clear batch
                batchFrames.clear();
            } else {
                continue;  // Wait until we have enough frames for the batch
            }
        }
        
        // Motion detection
        if (detectMotion) {
            processor.detectMotion(motionMask);
            cv::cvtColor(motionMask, motionMask, cv::COLOR_GRAY2BGR);
            cv::hconcat(outputFrame, motionMask, outputFrame);
        }
        
        // Optical flow
        if (opticalFlow) {
            processor.computeOpticalFlow(flowImage);
            if (!flowImage.empty()) {
                cv::hconcat(outputFrame, flowImage, outputFrame);
            }
        }
        
        // Object detection
        if (detectObjects) {
            processor.detectObjects(objectMask);
            if (!objectMask.empty()) {
                cv::cvtColor(objectMask, objectMask, cv::COLOR_GRAY2BGR);
                cv::hconcat(outputFrame, objectMask, outputFrame);
            }
        }
        
        // Write output frame
        if (!output.empty() && batchSize == 1) {
            processor.writeFrame(outputFrame);
        }
        
        // Display frame
        cv::imshow("CUDA Video Processor", outputFrame);
        
        // Process keyboard input
        int key = cv::waitKey(1);
        if (key == 27) { // ESC key
            isRunning = false;
        }
        
        frameCount++;
    }
    
    // Calculate and print processing time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    double fps = frameCount / (duration / 1000.0);
    
    std::cout << "Processing completed:" << std::endl;
    std::cout << "  Total frames: " << frameCount << std::endl;
    std::cout << "  Processing time: " << duration / 1000.0 << " seconds" << std::endl;
    std::cout << "  Average FPS: " << fps << std::endl;
    
    // Cleanup
    processor.close();
    cv::destroyAllWindows();
    
    return 0;
} 