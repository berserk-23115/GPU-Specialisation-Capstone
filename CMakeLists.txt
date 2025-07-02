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

# Find packages - specify required OpenCV components
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add CUDA source files (no OpenCV includes)
set(CUDA_SOURCES
    src/main.cu
    src/video_io.cu
    src/kernels.cu
    src/cuda_utils.cu
)

# Add C++ source files (with OpenCV includes)
set(CXX_SOURCES
    src/video_io_impl.cpp
)

# Create executable
add_executable(video_processor ${CUDA_SOURCES} ${CXX_SOURCES})

# Link libraries
target_link_libraries(video_processor ${OpenCV_LIBS})

# Set properties for CUDA
set_target_properties(video_processor PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Additional compiler definitions to resolve compatibility issues
# Only apply old ABI to CUDA files, let C++ files use default ABI
set_source_files_properties(${CUDA_SOURCES} PROPERTIES 
    COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=0"
)

# Ensure C++ files can link with OpenCV properly
set_target_properties(video_processor PROPERTIES
    LINKER_LANGUAGE CXX
)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Print debug information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}") 