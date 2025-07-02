#ifndef VIDEO_IO_H
#define VIDEO_IO_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include "kernels.h"

// Class for video input/output operations
class VideoProcessor {
public:
    VideoProcessor();
    ~VideoProcessor();

    // Open a video file or camera
    bool openVideo(const std::string& source, bool isFile = true);
    
    // Open output video file
    bool openOutputVideo(const std::string& filename, int codec = -1, double fps = -1.0);
    
    // Check if video is open
    bool isVideoOpen() const;
    
    // Get video properties
    int getWidth() const;
    int getHeight() const;
    int getChannels() const;
    double getFPS() const;
    int getTotalFrames() const;
    
    // Get current frame number
    int getCurrentFrame() const;
    
    // Read a frame from video
    bool readFrame(cv::Mat& frame);
    
    // Write frame to output video
    bool writeFrame(const cv::Mat& frame);
    
    // Close video
    void close();
    
    // Process a frame using CUDA
    bool processFrame(
        const cv::Mat& inputFrame, 
        cv::Mat& outputFrame,
        FilterType filterType,
        const FilterParams& filterParams,
        TransformType transformType = TransformType::NONE,
        const void* transformParams = nullptr
    );
    
    // Process batch of frames using CUDA
    bool processBatch(
        const std::vector<cv::Mat>& inputFrames,
        std::vector<cv::Mat>& outputFrames,
        FilterType filterType,
        const FilterParams& filterParams,
        TransformType transformType = TransformType::NONE,
        const void* transformParams = nullptr
    );
    
    // Detect motion between consecutive frames
    bool detectMotion(cv::Mat& motionMask, float threshold = 30.0f);
    
    // Compute optical flow between consecutive frames
    bool computeOpticalFlow(cv::Mat& flowImage);
    
    // Generate object detection mask
    bool detectObjects(cv::Mat& objectMask, float threshold = 0.5f);

private:
    // Video capture and writer
    cv::VideoCapture videoCapture;
    cv::VideoWriter videoWriter;
    
    // Video properties
    int width;
    int height;
    int channels;
    double fps;
    int totalFrames;
    int currentFrame;
    
    // Previous frame for motion detection
    cv::Mat previousFrame;
    
    // Convert cv::Mat to CUDA-compatible format
    unsigned char* matToCudaImage(const cv::Mat& frame);
    
    // Convert CUDA image back to cv::Mat
    void cudaImageToMat(unsigned char* cudaImage, cv::Mat& frame, int width, int height, int channels);
};

#endif // VIDEO_IO_H 