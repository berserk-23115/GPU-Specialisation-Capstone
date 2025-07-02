# CUDA Real-time Video Processor

A GPU-accelerated application for real-time video processing, enhancement, and analysis using CUDA.

## Overview

This capstone project demonstrates the power of GPU computing for real-time video processing. The application leverages CUDA to perform computationally intensive operations on video frames in parallel, achieving significant performance improvements compared to CPU-based processing.

Key features:

- Multiple GPU-accelerated image filters (blur, sharpen, edge detection, emboss, sepia, etc.)
- Real-time video processing from files or camera input
- Advanced visual effects like thermal vision and night vision simulation
- Motion detection between consecutive frames
- Optical flow visualization
- Simple object detection
- Batch processing for higher throughput
- Performance benchmarking tools

## Requirements

- NVIDIA CUDA Toolkit (11.0+)
- OpenCV 4.x
- C++17 compatible compiler
- CMake 3.10+

## Installation

### Option 1: Building from Source

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd GPU-Specialisation-Capstone
   ```

2. Create a build directory and compile:
   ```bash
   ./build.sh
   ```
   
   Or manually:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make -j$(nproc)
   ```

### Option 2: Using Docker

A Dockerfile is provided for containerized development and execution:

```bash
docker build -t cuda-video-processor .
docker run --gpus all -it --rm cuda-video-processor
```

## Usage

The application supports various command-line arguments:

```bash
./video_processor [options]
```

### Options

- `--input <source>`: Input source (video file path or camera index)
- `--output <filename>`: Output video file (optional)
- `--filter <filter_type>`: Filter to apply (default: none)
- `--transform <transform>`: Transform to apply (default: none)
- `--intensity <value>`: Filter intensity (0.0-1.0, default: 0.5)
- `--detect-motion`: Enable motion detection
- `--optical-flow`: Enable optical flow visualization
- `--detect-objects`: Enable simple object detection
- `--benchmark`: Run performance benchmark
- `--batch-size <size>`: Batch processing size (default: 1)
- `--help`: Display help message

### Available Filters

- `none`: No filter
- `blur`: Gaussian blur
- `sharpen`: Sharpen effect
- `edge_detect`: Edge detection
- `emboss`: Emboss effect
- `sepia`: Sepia tone
- `grayscale`: Grayscale conversion
- `negative`: Color inversion
- `cartoon`: Cartoon effect
- `sketch`: Sketch effect
- `night_vision`: Night vision effect
- `thermal`: Thermal vision effect

### Available Transformations

- `none`: No transformation
- `rotate_90`: Rotate image by 90 degrees
- `rotate_180`: Rotate image by 180 degrees
- `rotate_270`: Rotate image by 270 degrees
- `flip_h`: Flip horizontally
- `flip_v`: Flip vertically

### Examples

Process a video file with a blur filter:
```bash
./video_processor --input input_video.mp4 --output output_video.mp4 --filter blur --intensity 0.7
```

Use camera input with thermal vision effect:
```bash
./video_processor --input 0 --filter thermal --intensity 0.8
```

Enable motion detection with edge detection filter:
```bash
./video_processor --input input_video.mp4 --filter edge_detect --detect-motion
```

Run performance benchmark:
```bash
./video_processor --input input_video.mp4 --benchmark
```

## How it Works

### CUDA-Accelerated Processing Pipeline

1. **Frame Acquisition**: Frames are captured from video files or camera.
2. **GPU Transfer**: Frame data is transferred to the GPU memory.
3. **Parallel Processing**: Each pixel is processed in parallel using thousands of CUDA threads.
4. **Filter Application**: Mathematical operations are applied to transform pixel values.
5. **Result Transfer**: Processed frames are transferred back to CPU memory.
6. **Display/Storage**: Results are displayed in real-time and/or saved to disk.

### CUDA Kernels

The project implements various CUDA kernels for different image processing tasks:

- **Convolution Kernel**: Applies convolution filters (blur, sharpen, edge detection, etc.)
- **Color Transformation Kernels**: Applies specialized color effects (sepia, grayscale, etc.)
- **Motion Detection Kernel**: Computes differences between consecutive frames
- **Special Effect Kernels**: Implements complex effects like thermal vision and night vision

## Performance

The GPU acceleration provides significant performance improvements over CPU-based implementations:

- Processing HD video (1920x1080) in real-time at 30+ FPS
- 10-20x speedup compared to equivalent CPU implementation
- Efficient batch processing for higher throughput

Actual performance depends on GPU specifications, filter complexity, and frame resolution.

## Future Improvements

- Advanced object detection and tracking using CUDA-accelerated ML models
- Deep learning-based video enhancement
- Support for multiple GPU devices
- Hardware-accelerated video encoding/decoding
- More complex visual effects and transformations

## License

[MIT License](LICENSE) 