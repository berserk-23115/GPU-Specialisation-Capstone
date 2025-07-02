#!/bin/bash

# Exit on error
set -e

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure and build the project
echo "Configuring project with CMake..."
cmake ..

echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Return to original directory
cd ..

echo "Build completed successfully."
echo "The executable is located at: $(pwd)/build/video_processor" 