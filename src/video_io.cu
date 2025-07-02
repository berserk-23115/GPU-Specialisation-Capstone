#include "video_io.h"
#include "cuda_utils.h"
#include "kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// C++ implementation functions (defined in video_io_impl.cpp)
extern "C" {
    void* create_opencv_processor();
    void destroy_opencv_processor(void* processor);
    bool open_opencv_video(void* processor, const char* source, bool isFile);
    bool open_opencv_output(void* processor, const char* filename);
    bool is_opencv_video_open(void* processor);
    int get_opencv_width(void* processor);
    int get_opencv_height(void* processor);
    int get_opencv_channels(void* processor);
    double get_opencv_fps(void* processor);
    int get_opencv_total_frames(void* processor);
    int get_opencv_current_frame(void* processor);
    bool read_opencv_frame(void* processor);
    bool write_opencv_frame(void* processor);
    void close_opencv_video(void* processor);
    
    // Frame data access
    unsigned char* get_current_frame_data(void* processor);
    unsigned char* get_output_frame_data(void* processor);
    void set_output_frame_data(void* processor, unsigned char* data);
}

// VideoProcessor structure
struct VideoProcessor {
    void* opencv_processor;  // Opaque pointer to OpenCV implementation
    unsigned char* current_frame_data;
    unsigned char* output_frame_data;
    int width;
    int height;
    int channels;
    size_t frame_size;
    
    // CUDA memory buffers
    unsigned char* d_input_frame;
    unsigned char* d_output_frame;
};

// Create video processor
VideoProcessor* createVideoProcessor() {
    VideoProcessor* processor = (VideoProcessor*)malloc(sizeof(VideoProcessor));
    if (!processor) {
        printf("Error: Could not allocate VideoProcessor\n");
        return NULL;
    }
    
    // Initialize members
    processor->opencv_processor = create_opencv_processor();
    processor->current_frame_data = NULL;
    processor->output_frame_data = NULL;
    processor->width = 0;
    processor->height = 0;
    processor->channels = 0;
    processor->frame_size = 0;
    processor->d_input_frame = NULL;
    processor->d_output_frame = NULL;
    
    if (!processor->opencv_processor) {
        printf("Error: Could not create OpenCV processor\n");
        free(processor);
        return NULL;
    }
    
    return processor;
}

// Destroy video processor
void destroyVideoProcessor(VideoProcessor* processor) {
    if (!processor) return;
    
    // Clean up CUDA memory
    if (processor->d_input_frame) {
        cudaFree(processor->d_input_frame);
    }
    if (processor->d_output_frame) {
        cudaFree(processor->d_output_frame);
    }
    
    // Clean up host memory
    if (processor->current_frame_data) {
        free(processor->current_frame_data);
    }
    if (processor->output_frame_data) {
        free(processor->output_frame_data);
    }
    
    // Clean up OpenCV processor
    destroy_opencv_processor(processor->opencv_processor);
    
    free(processor);
}

// Open video source
bool openVideoSource(VideoProcessor* processor, const char* source, bool isFile) {
    if (!processor || !processor->opencv_processor) {
        return false;
    }
    
    if (!open_opencv_video(processor->opencv_processor, source, isFile)) {
        return false;
    }
    
    // Get video properties
    processor->width = get_opencv_width(processor->opencv_processor);
    processor->height = get_opencv_height(processor->opencv_processor);
    processor->channels = get_opencv_channels(processor->opencv_processor);
    processor->frame_size = processor->width * processor->height * processor->channels;
    
    // Allocate host memory for frame data
    processor->current_frame_data = (unsigned char*)malloc(processor->frame_size);
    processor->output_frame_data = (unsigned char*)malloc(processor->frame_size);
    
    if (!processor->current_frame_data || !processor->output_frame_data) {
        printf("Error: Could not allocate frame memory\n");
        return false;
    }
    
    // Allocate CUDA memory
    CUDA_CHECK_ERROR(cudaMalloc((void**)&processor->d_input_frame, processor->frame_size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&processor->d_output_frame, processor->frame_size));
    
    return true;
}

// Open output video
bool openVideoOutput(VideoProcessor* processor, const char* filename) {
    if (!processor || !processor->opencv_processor) {
        return false;
    }
    
    return open_opencv_output(processor->opencv_processor, filename);
}

// Check if video is open
bool isVideoOpen(VideoProcessor* processor) {
    if (!processor || !processor->opencv_processor) {
        return false;
    }
    
    return is_opencv_video_open(processor->opencv_processor);
}

// Get video properties
int getVideoWidth(VideoProcessor* processor) {
    return processor ? processor->width : 0;
}

int getVideoHeight(VideoProcessor* processor) {
    return processor ? processor->height : 0;
}

int getVideoChannels(VideoProcessor* processor) {
    return processor ? processor->channels : 0;
}

double getVideoFPS(VideoProcessor* processor) {
    if (!processor || !processor->opencv_processor) {
        return 0.0;
    }
    
    return get_opencv_fps(processor->opencv_processor);
}

int getVideoTotalFrames(VideoProcessor* processor) {
    if (!processor || !processor->opencv_processor) {
        return 0;
    }
    
    return get_opencv_total_frames(processor->opencv_processor);
}

int getVideoCurrentFrame(VideoProcessor* processor) {
    if (!processor || !processor->opencv_processor) {
        return 0;
    }
    
    return get_opencv_current_frame(processor->opencv_processor);
}

// Read video frame
bool readVideoFrame(VideoProcessor* processor) {
    if (!processor || !processor->opencv_processor) {
        return false;
    }
    
    if (!read_opencv_frame(processor->opencv_processor)) {
        return false;
    }
    
    // Get frame data from OpenCV
    unsigned char* frame_data = get_current_frame_data(processor->opencv_processor);
    if (!frame_data) {
        return false;
    }
    
    // Copy to our buffer
    memcpy(processor->current_frame_data, frame_data, processor->frame_size);
    
    return true;
}

// Write video frame
bool writeVideoFrame(VideoProcessor* processor) {
    if (!processor || !processor->opencv_processor) {
        return false;
    }
    
    // Set output frame data in OpenCV processor
    set_output_frame_data(processor->opencv_processor, processor->output_frame_data);
    
    return write_opencv_frame(processor->opencv_processor);
}

// Process video frame with CUDA
bool processVideoFrame(
    VideoProcessor* processor,
    FilterType filterType,
    const FilterParams& filterParams,
    TransformType transformType
) {
    if (!processor || !processor->current_frame_data) {
        return false;
    }
    
    // Copy input frame to GPU
    CUDA_CHECK_ERROR(cudaMemcpy(processor->d_input_frame, processor->current_frame_data, 
                                processor->frame_size, cudaMemcpyHostToDevice));
    
    // Apply transformation first if needed
    if (transformType != TransformType::NONE) {
        applyTransformation(
            processor->d_input_frame,
            processor->d_output_frame,
            transformType,
            processor->width,
            processor->height,
            processor->channels
        );
        
        // Swap buffers
        unsigned char* temp = processor->d_input_frame;
        processor->d_input_frame = processor->d_output_frame;
        processor->d_output_frame = temp;
    }
    
    // Apply filter if needed
    if (filterType != FilterType::NONE) {
        applySpecialFilter(
            processor->d_input_frame,
            processor->d_output_frame,
            filterType,
            filterParams,
            processor->width,
            processor->height,
            processor->channels
        );
    } else {
        // No filter, just copy input to output
        CUDA_CHECK_ERROR(cudaMemcpy(processor->d_output_frame, processor->d_input_frame, 
                                    processor->frame_size, cudaMemcpyDeviceToDevice));
    }
    
    // Copy result back to host
    CUDA_CHECK_ERROR(cudaMemcpy(processor->output_frame_data, processor->d_output_frame, 
                                processor->frame_size, cudaMemcpyDeviceToHost));
    
    return true;
}

// Close video
void closeVideo(VideoProcessor* processor) {
    if (!processor || !processor->opencv_processor) {
        return;
    }
    
    close_opencv_video(processor->opencv_processor);
}
