#ifndef VIDEO_IO_H
#define VIDEO_IO_H

#include "kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for opaque VideoProcessor handle
typedef struct VideoProcessor VideoProcessor;

// Create and destroy video processor
VideoProcessor* createVideoProcessor();
void destroyVideoProcessor(VideoProcessor* processor);

// Open video source (file or camera)
bool openVideoSource(VideoProcessor* processor, const char* source, bool isFile);

// Open output video file
bool openVideoOutput(VideoProcessor* processor, const char* filename);

// Check if video is open
bool isVideoOpen(VideoProcessor* processor);

// Get video properties
int getVideoWidth(VideoProcessor* processor);
int getVideoHeight(VideoProcessor* processor);
int getVideoChannels(VideoProcessor* processor);
double getVideoFPS(VideoProcessor* processor);
int getVideoTotalFrames(VideoProcessor* processor);
int getVideoCurrentFrame(VideoProcessor* processor);

// Read and write frames
bool readVideoFrame(VideoProcessor* processor);
bool writeVideoFrame(VideoProcessor* processor);

// Process frame with CUDA
bool processVideoFrame(
    VideoProcessor* processor,
    FilterType filterType,
    const FilterParams& filterParams,
    TransformType transformType
);

// Close video
void closeVideo(VideoProcessor* processor);

#ifdef __cplusplus
}
#endif

#endif // VIDEO_IO_H 