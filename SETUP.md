# CUDA Video Processor Setup Guide

This document provides detailed instructions for setting up and running the CUDA Video Processor application.

## System Requirements

- NVIDIA GPU with CUDA support (Compute Capability 5.0 or higher recommended)
- NVIDIA CUDA Toolkit 11.0 or newer
- OpenCV 4.x with CUDA support
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 19.14+)
- CMake 3.10 or newer

## Installation Steps

### 1. Install CUDA Toolkit

#### Ubuntu/Debian:
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Install CUDA Toolkit
sudo apt update
sudo apt install cuda-toolkit-11-8
```

#### macOS:
Download and install from: https://developer.nvidia.com/cuda-downloads

#### Windows:
Download and install from: https://developer.nvidia.com/cuda-downloads

### 2. Install OpenCV with CUDA Support

#### Ubuntu/Debian:
```bash
# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-dev

# Clone OpenCV and OpenCV contrib repositories
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Set up build
cd opencv
mkdir build && cd build

# Configure with CMake
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_GTK=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF ..

# Compile and install
make -j$(nproc)
sudo make install
sudo ldconfig
```

#### macOS:
```bash
# Install with Homebrew
brew install opencv

# Note: The Homebrew version may not include CUDA support.
# For CUDA support, you'll need to build from source similar to the Ubuntu instructions.
```

#### Windows:
Download pre-built binaries from: https://opencv.org/releases/
or build from source following OpenCV documentation.

### 3. Clone and Build the Project

```bash
# Clone this repository
git clone <repository-url>
cd GPU-Specialisation-Capstone

# Build using the provided script
./build.sh

# Or manually:
mkdir -p build
cd build
cmake ..
make -j$(nproc)  # On Windows, use: cmake --build . --config Release
```

## Using Docker (Alternative)

If you prefer using Docker, a Dockerfile is provided:

```bash
# Build the Docker image
docker build -t cuda-video-processor .

# Run the container with GPU support
docker run --gpus all -it --rm cuda-video-processor --help

# To process a video file, mount a volume containing your video:
docker run --gpus all -it --rm -v /path/to/videos:/videos cuda-video-processor --input /videos/input.mp4 --output /videos/output.mp4 --filter thermal
```

## Verifying Installation

To verify that the application is working correctly:

1. Run the application with the help flag:
   ```bash
   ./build/video_processor --help
   ```

2. Run the benchmark to test CUDA functionality:
   ```bash
   ./build/video_processor --benchmark
   ```

3. Process a sample video (or use camera input):
   ```bash
   ./build/video_processor --input /path/to/video.mp4 --filter blur
   ```

## Troubleshooting

### CUDA Issues

- **Error: No CUDA devices found**
  - Verify NVIDIA drivers are installed: `nvidia-smi`
  - Check CUDA installation: `nvcc --version`

- **CUDA version mismatch**
  - Ensure the CUDA Toolkit version matches the one used to build OpenCV

### OpenCV Issues

- **Cannot find OpenCV libraries**
  - Ensure OpenCV is correctly installed: `pkg-config --modversion opencv4`
  - Check library path: `echo $LD_LIBRARY_PATH`

- **Runtime error about missing GPU functions**
  - Verify OpenCV was built with CUDA support: `opencv_version --verbose`

### Build Issues

- **CMake cannot find CUDA or OpenCV**
  - Set environment variables:
    ```bash
    export OpenCV_DIR=/path/to/opencv/installation
    export CUDA_HOME=/usr/local/cuda
    ```

- **Compilation errors**
  - Check C++ compiler version: `g++ --version`
  - Ensure you're using C++17 or later 