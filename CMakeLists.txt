cmake_minimum_required(VERSION 3.10)
project(CUDAVideoProcessor LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17 -O3)

# Add CUDA source files
file(GLOB SOURCES "src/*.cu" "src/*.cpp")

# Create executable
cuda_add_executable(video_processor ${SOURCES})

# Link libraries
target_link_libraries(video_processor ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Print OpenCV version
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}") 