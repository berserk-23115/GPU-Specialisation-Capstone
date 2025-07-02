cmake_minimum_required(VERSION 3.18)
project(CUDAVideoProcessor LANGUAGES CXX CUDA)

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

# Set C++ standard for CXX files (e.g., .cpp)
set_target_properties(video_processor PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Set compile options for CUDA files, including for the host compiler
target_compile_options(video_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -std=c++14                 # For nvcc's device compiler
        -Xcompiler "-std=c++14"    # Pass -std=c++14 to the host compiler
    >
)

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