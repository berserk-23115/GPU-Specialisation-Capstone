#include "video_io.h"
#include "cuda_utils.h"
#include <iostream>

// Constructor
VideoProcessor::VideoProcessor() 
    : width(0), height(0), channels(0), fps(0.0), totalFrames(0), currentFrame(0) {
}

// Destructor
VideoProcessor::~VideoProcessor() {
    close();
}

// Open a video file or camera
bool VideoProcessor::openVideo(const std::string& source, bool isFile) {
    // Close any open video
    close();
    
    try {
        // Open video source
        if (isFile) {
            // Open video file
            if (!videoCapture.open(source)) {
                std::cerr << "Error: Could not open video file: " << source << std::endl;
                return false;
            }
        } else {
            // Open camera
            int cameraIndex = std::stoi(source);
            if (!videoCapture.open(cameraIndex)) {
                std::cerr << "Error: Could not open camera: " << cameraIndex << std::endl;
                return false;
            }
        }
        
        // Get video properties
        width = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = videoCapture.get(cv::CAP_PROP_FPS);
        totalFrames = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_COUNT));
        currentFrame = 0;
        
        // Read a frame to determine number of channels
        cv::Mat frame;
        if (videoCapture.read(frame)) {
            channels = frame.channels();
        } else {
            channels = 3;  // Default to 3 channels
        }
        
        // Reset video to beginning
        if (isFile) {
            videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error opening video source: " << e.what() << std::endl;
        return false;
    }
}

// Open output video file
bool VideoProcessor::openOutputVideo(const std::string& filename, int codec, double fps) {
    try {
        // Get four character code for video codec
        int fourcc;
        if (codec == -1) {
            // Auto-detect from filename extension
            std::string ext = filename.substr(filename.find_last_of('.') + 1);
            if (ext == "mp4") {
                fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            } else if (ext == "avi") {
                fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            } else {
                fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            }
        } else {
            fourcc = codec;
        }
        
        // Use input video's FPS if not specified
        if (fps < 0) {
            fps = this->fps;
        }
        
        // Ensure valid FPS
        if (fps <= 0) {
            fps = 30.0;
        }
        
        // Create video writer
        return videoWriter.open(filename, fourcc, fps, cv::Size(width, height), true);
    } catch (const std::exception& e) {
        std::cerr << "Error opening output video: " << e.what() << std::endl;
        return false;
    }
}

// Check if video is open
bool VideoProcessor::isVideoOpen() const {
    return videoCapture.isOpened();
}

// Get video properties
int VideoProcessor::getWidth() const { return width; }
int VideoProcessor::getHeight() const { return height; }
int VideoProcessor::getChannels() const { return channels; }
double VideoProcessor::getFPS() const { return fps; }
int VideoProcessor::getTotalFrames() const { return totalFrames; }

// Get current frame number
int VideoProcessor::getCurrentFrame() const {
    return currentFrame;
}

// Read a frame from video
bool VideoProcessor::readFrame(cv::Mat& frame) {
    if (!videoCapture.isOpened()) {
        return false;
    }
    
    // Read frame
    bool success = videoCapture.read(frame);
    
    if (success) {
        currentFrame++;
        
        // Store frame for motion detection
        if (previousFrame.empty() || previousFrame.size() != frame.size()) {
            previousFrame = frame.clone();
        } else {
            frame.copyTo(previousFrame);
        }
    }
    
    return success;
}

// Write frame to output video
bool VideoProcessor::writeFrame(const cv::Mat& frame) {
    if (!videoWriter.isOpened()) {
        return false;
    }
    
    // Write frame
    videoWriter.write(frame);
    return true;
}

// Close video
void VideoProcessor::close() {
    if (videoCapture.isOpened()) {
        videoCapture.release();
    }
    
    if (videoWriter.isOpened()) {
        videoWriter.release();
    }
    
    // Reset properties
    width = 0;
    height = 0;
    channels = 0;
    fps = 0.0;
    totalFrames = 0;
    currentFrame = 0;
    
    // Clear previous frame
    previousFrame.release();
}

// Process a frame using CUDA
bool VideoProcessor::processFrame(
    const cv::Mat& inputFrame, 
    cv::Mat& outputFrame,
    FilterType filterType,
    const FilterParams& filterParams,
    TransformType transformType,
    const void* transformParams
) {
    try {
        // Ensure output frame has same size and type as input
        if (outputFrame.size() != inputFrame.size() || outputFrame.type() != inputFrame.type()) {
            outputFrame = cv::Mat(inputFrame.size(), inputFrame.type());
        }
        
        // If no processing is required, just copy the frame
        if (filterType == FilterType::NONE && transformType == TransformType::NONE) {
            inputFrame.copyTo(outputFrame);
            return true;
        }
        
        // Convert to CUDA-compatible format
        unsigned char* d_inputImage = matToCudaImage(inputFrame);
        unsigned char* d_outputImage = new unsigned char[inputFrame.total() * inputFrame.channels()];
        
        // Apply transformation first if requested
        if (transformType != TransformType::NONE) {
            // Apply transformation using CUDA
            applyTransformation(
                d_inputImage,
                d_outputImage,
                transformType,
                inputFrame.cols,
                inputFrame.rows,
                inputFrame.channels(),
                transformParams
            );
            
            // Swap pointers so output becomes input for filter operation
            unsigned char* temp = d_inputImage;
            d_inputImage = d_outputImage;
            d_outputImage = temp;
        }
        
        // Apply filter if requested
        if (filterType != FilterType::NONE) {
            if (filterType == FilterType::BLUR || filterType == FilterType::SHARPEN || 
                filterType == FilterType::EDGE_DETECT || filterType == FilterType::EMBOSS) {
                // Generate filter kernel
                int filterWidth = 3;  // Using 3x3 filter
                float h_filter[9];
                generateFilter(h_filter, filterWidth, filterType, filterParams);
                
                // Apply convolution
                applyConvolution(
                    d_inputImage,
                    d_outputImage,
                    h_filter,
                    filterWidth,
                    inputFrame.cols,
                    inputFrame.rows,
                    inputFrame.channels()
                );
            } else {
                // Apply specialized filter
                applySpecialFilter(
                    d_inputImage,
                    d_outputImage,
                    filterType,
                    filterParams,
                    inputFrame.cols,
                    inputFrame.rows,
                    inputFrame.channels()
                );
            }
        }
        
        // Convert CUDA image back to cv::Mat
        cudaImageToMat(d_outputImage, outputFrame, inputFrame.cols, inputFrame.rows, inputFrame.channels());
        
        // Free CUDA memory
        delete[] d_inputImage;
        delete[] d_outputImage;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error processing frame: " << e.what() << std::endl;
        return false;
    }
}

// Process batch of frames using CUDA
bool VideoProcessor::processBatch(
    const std::vector<cv::Mat>& inputFrames,
    std::vector<cv::Mat>& outputFrames,
    FilterType filterType,
    const FilterParams& filterParams,
    TransformType transformType,
    const void* transformParams
) {
    // Check if input batch is empty
    if (inputFrames.empty()) {
        return false;
    }
    
    try {
        // Prepare output frames vector
        outputFrames.resize(inputFrames.size());
        
        // Process each frame in the batch
        // Note: In a more optimized implementation, we would batch process on the GPU
        // rather than sending each frame separately
        for (size_t i = 0; i < inputFrames.size(); i++) {
            if (!processFrame(
                inputFrames[i], 
                outputFrames[i], 
                filterType, 
                filterParams, 
                transformType, 
                transformParams)) {
                return false;
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error processing batch: " << e.what() << std::endl;
        return false;
    }
}

// Detect motion between consecutive frames
bool VideoProcessor::detectMotion(cv::Mat& motionMask, float threshold) {
    // Check if we have a previous frame
    if (previousFrame.empty()) {
        return false;
    }
    
    try {
        // Create motion mask with same size as frames
        motionMask = cv::Mat(previousFrame.size(), CV_8UC1, cv::Scalar(0));
        
        // Convert frames to CUDA format
        unsigned char* d_prevFrame = matToCudaImage(previousFrame);
        unsigned char* d_currFrame = matToCudaImage(previousFrame);  // Using same frame for placeholder
        unsigned char* d_motionMask = new unsigned char[previousFrame.rows * previousFrame.cols];
        
        // Detect motion using CUDA
        detectMotion(
            d_prevFrame,
            d_currFrame,
            d_motionMask,
            previousFrame.cols,
            previousFrame.rows,
            threshold
        );
        
        // Convert motion mask back to cv::Mat
        cv::Mat temp(previousFrame.rows, previousFrame.cols, CV_8UC1);
        std::memcpy(temp.data, d_motionMask, previousFrame.rows * previousFrame.cols);
        motionMask = temp;
        
        // Free CUDA memory
        delete[] d_prevFrame;
        delete[] d_currFrame;
        delete[] d_motionMask;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error detecting motion: " << e.what() << std::endl;
        return false;
    }
}

// Compute optical flow between consecutive frames
bool VideoProcessor::computeOpticalFlow(cv::Mat& flowImage) {
    // Check if we have a previous frame
    if (previousFrame.empty()) {
        return false;
    }
    
    try {
        // Create optical flow image with same size as frames
        flowImage = cv::Mat(previousFrame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        
        // Compute optical flow using CUDA
        // Note: This is a placeholder. In a real implementation, we would compute optical flow
        // using a CUDA kernel
        
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(previousFrame, previousFrame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        
        // Visualize the flow
        for (int y = 0; y < previousFrame.rows; y += 10) {
            for (int x = 0; x < previousFrame.cols; x += 10) {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                cv::line(flowImage, cv::Point(x, y), 
                        cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                cv::circle(flowImage, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error computing optical flow: " << e.what() << std::endl;
        return false;
    }
}

// Generate object detection mask
bool VideoProcessor::detectObjects(cv::Mat& objectMask, float threshold) {
    // Check if we have a frame
    if (previousFrame.empty()) {
        return false;
    }
    
    try {
        // Create object mask with same size as frames
        objectMask = cv::Mat(previousFrame.size(), CV_8UC1, cv::Scalar(0));
        
        // Convert frame to CUDA format
        unsigned char* d_frame = matToCudaImage(previousFrame);
        unsigned char* d_objectMask = new unsigned char[previousFrame.rows * previousFrame.cols];
        
        // Detect objects using CUDA
        generateObjectMask(
            d_frame,
            d_objectMask,
            previousFrame.cols,
            previousFrame.rows,
            threshold
        );
        
        // Convert object mask back to cv::Mat
        cv::Mat temp(previousFrame.rows, previousFrame.cols, CV_8UC1);
        std::memcpy(temp.data, d_objectMask, previousFrame.rows * previousFrame.cols);
        objectMask = temp;
        
        // Free CUDA memory
        delete[] d_frame;
        delete[] d_objectMask;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error detecting objects: " << e.what() << std::endl;
        return false;
    }
}

// Convert cv::Mat to CUDA-compatible format
unsigned char* VideoProcessor::matToCudaImage(const cv::Mat& frame) {
    size_t size = frame.total() * frame.channels();
    unsigned char* image = new unsigned char[size];
    std::memcpy(image, frame.data, size);
    return image;
}

// Convert CUDA image back to cv::Mat
void VideoProcessor::cudaImageToMat(unsigned char* cudaImage, cv::Mat& frame, int width, int height, int channels) {
    // Create cv::Mat
    frame = cv::Mat(height, width, CV_8UC(channels));
    
    // Copy data
    std::memcpy(frame.data, cudaImage, width * height * channels);
}
