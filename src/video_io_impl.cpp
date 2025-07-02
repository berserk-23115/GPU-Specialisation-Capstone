#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <cstring>

// Internal OpenCV processor class
class OpenCVProcessor {
public:
    cv::VideoCapture videoCapture;
    cv::VideoWriter videoWriter;
    cv::Mat currentFrame;
    cv::Mat outputFrame;
    int width;
    int height;
    int channels;
    double fps;
    int totalFrames;
    int currentFrameNum;
    
    OpenCVProcessor() : width(0), height(0), channels(0), fps(0.0), totalFrames(0), currentFrameNum(0) {}
    
    ~OpenCVProcessor() {
        close();
    }
    
    void close() {
        if (videoCapture.isOpened()) {
            videoCapture.release();
        }
        if (videoWriter.isOpened()) {
            videoWriter.release();
        }
        currentFrame.release();
        outputFrame.release();
    }
};

// C interface functions
extern "C" {

void* create_opencv_processor() {
    try {
        return new OpenCVProcessor();
    } catch (...) {
        return nullptr;
    }
}

void destroy_opencv_processor(void* processor) {
    if (processor) {
        delete static_cast<OpenCVProcessor*>(processor);
    }
}

bool open_opencv_video(void* processor, const char* source, bool isFile) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    if (!proc) return false;
    
    try {
        proc->close();
        
        if (isFile) {
            if (!proc->videoCapture.open(std::string(source))) {
                std::cerr << "Error: Could not open video file: " << source << std::endl;
                return false;
            }
        } else {
            int cameraIndex = std::atoi(source);
            if (!proc->videoCapture.open(cameraIndex)) {
                std::cerr << "Error: Could not open camera: " << cameraIndex << std::endl;
                return false;
            }
        }
        
        // Get video properties
        proc->width = static_cast<int>(proc->videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
        proc->height = static_cast<int>(proc->videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
        proc->fps = proc->videoCapture.get(cv::CAP_PROP_FPS);
        proc->totalFrames = static_cast<int>(proc->videoCapture.get(cv::CAP_PROP_FRAME_COUNT));
        proc->currentFrameNum = 0;
        
        // Read a frame to determine number of channels
        cv::Mat frame;
        if (proc->videoCapture.read(frame)) {
            proc->channels = frame.channels();
            // Reset video to beginning
            if (isFile) {
                proc->videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
            }
        } else {
            proc->channels = 3;  // Default to 3 channels
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error opening video source: " << e.what() << std::endl;
        return false;
    }
}

bool open_opencv_output(void* processor, const char* filename) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    if (!proc) return false;
    
    try {
        std::string filenameStr(filename);
        std::string ext = filenameStr.substr(filenameStr.find_last_of('.') + 1);
        
        int fourcc;
        if (ext == "mp4") {
            fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        } else if (ext == "avi") {
            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        } else {
            fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        }
        
        double fps = proc->fps > 0 ? proc->fps : 30.0;
        
        return proc->videoWriter.open(filenameStr, fourcc, fps, 
                                     cv::Size(proc->width, proc->height), true);
    } catch (const std::exception& e) {
        std::cerr << "Error opening output video: " << e.what() << std::endl;
        return false;
    }
}

bool is_opencv_video_open(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    return proc ? proc->videoCapture.isOpened() : false;
}

int get_opencv_width(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    return proc ? proc->width : 0;
}

int get_opencv_height(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    return proc ? proc->height : 0;
}

int get_opencv_channels(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    return proc ? proc->channels : 0;
}

double get_opencv_fps(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    return proc ? proc->fps : 0.0;
}

int get_opencv_total_frames(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    return proc ? proc->totalFrames : 0;
}

int get_opencv_current_frame(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    return proc ? proc->currentFrameNum : 0;
}

bool read_opencv_frame(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    if (!proc || !proc->videoCapture.isOpened()) {
        return false;
    }
    
    bool success = proc->videoCapture.read(proc->currentFrame);
    if (success) {
        proc->currentFrameNum++;
    }
    return success;
}

bool write_opencv_frame(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    if (!proc || !proc->videoWriter.isOpened()) {
        return false;
    }
    
    if (proc->outputFrame.empty()) {
        return false;
    }
    
    proc->videoWriter.write(proc->outputFrame);
    return true;
}

void close_opencv_video(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    if (proc) {
        proc->close();
    }
}

unsigned char* get_current_frame_data(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    if (!proc || proc->currentFrame.empty()) {
        return nullptr;
    }
    return proc->currentFrame.data;
}

unsigned char* get_output_frame_data(void* processor) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    if (!proc || proc->outputFrame.empty()) {
        return nullptr;
    }
    return proc->outputFrame.data;
}

void set_output_frame_data(void* processor, unsigned char* data) {
    OpenCVProcessor* proc = static_cast<OpenCVProcessor*>(processor);
    if (!proc || !data) {
        return;
    }
    
    // Create output frame if it doesn't exist
    if (proc->outputFrame.empty()) {
        proc->outputFrame = cv::Mat(proc->height, proc->width, CV_8UC(proc->channels));
    }
    
    // Copy data to output frame
    size_t dataSize = proc->width * proc->height * proc->channels;
    std::memcpy(proc->outputFrame.data, data, dataSize);
}

} // extern "C" 