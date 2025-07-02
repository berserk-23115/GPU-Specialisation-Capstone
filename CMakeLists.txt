cmake_minimum_required(VERSION 3.18)
project(CUDAVideoProcessor LANGUAGES CXX CUDA)

# Set C++ standard to 14 (maximum supported by CUDA 11.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Fix for GCC 11.4.0 and CUDA 11.5.119 compatibility issues
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-std=c++14")

# Additional CUDA flags to fix std_function.h parameter pack expansion issues
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-gnu-unique")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=0")

# Disable problematic warnings that can cause compilation errors
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-w")

# Find packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set CUDA architecture (adjust based on your GPU)
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
    CUDA_SEPARABLE_COMPILATION ON
)

# Additional compiler definitions to resolve compatibility issues
target_compile_definitions(video_processor PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Print versions
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}") 