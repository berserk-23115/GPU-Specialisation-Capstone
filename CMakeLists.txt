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

# Set C++ and CUDA standard for the target
set_target_properties(video_processor PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)

# Set compiler flags for the target
target_compile_options(video_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
        -Xcudafe --diag_suppress=field_without_dll_interface
        -Xcudafe --diag_suppress=base_class_has_different_dll_interface
        -Xcudafe --diag_suppress=declared_but_not_referenced
        -std=c++14
    >
)

# Set properties for CUDA
set_target_properties(video_processor PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(video_processor ${OpenCV_LIBS})

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Print versions
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}") 