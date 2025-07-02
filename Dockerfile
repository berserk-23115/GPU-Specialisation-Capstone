FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and build OpenCV
WORKDIR /opt
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv && \
    git checkout 4.7.0 && \
    cd ../opencv_contrib && \
    git checkout 4.7.0 && \
    cd ../opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D CUDA_ARCH_BIN=7.5,8.0,8.6 \
          -D CUDA_ARCH_PTX="" \
          -D WITH_CUBLAS=ON \
          -D WITH_TBB=ON \
          -D WITH_V4L=ON \
          -D BUILD_opencv_python3=ON \
          -D INSTALL_PYTHON_EXAMPLES=OFF \
          -D BUILD_EXAMPLES=OFF .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Create application directory
WORKDIR /app

# Copy application files
COPY . /app/

# Build the application
RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)

# Set the entry point
ENTRYPOINT ["/app/build/video_processor"] 