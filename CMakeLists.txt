cmake_minimum_required(VERSION 3.18)
project(CUDAVideoProcessor LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly tell nvcc to use C++14 for its host compiler (gcc)
# This is the most critical fix.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14 -Xcompiler=-std=c++14")

# Find packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add source files
file(GLOB SOURCES "src/*.cu" "src/*.cpp")

# Create executable
add_executable(video_processor ${SOURCES})

# Link libraries
target_link_libraries(video_processor ${OpenCV_LIBS})

# Set properties for CUDA
set_target_properties(video_processor PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Print versions
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}") 