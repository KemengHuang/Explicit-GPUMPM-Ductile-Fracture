# =============================================================================
# Basic Configuration
# =============================================================================

# Minimum required CMake version
cmake_minimum_required(VERSION 3.18)

# Set policy to avoid compiler ID conflicts
cmake_policy(SET CMP0069 NEW)

# Project settings
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
set(CUDA_ARCHITECTURES 86)  # SM86 for Ampere architecture (RTX 30xx series)

# Initialize project with C++ and CUDA languages
project(gpu_mpm LANGUAGES CXX CUDA)

# =============================================================================
# Language Standards Configuration
# =============================================================================

# Set C++17 standard with strict compliance
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA 17 standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# =============================================================================
# Dependency Management
# =============================================================================

# Find CUDA Toolkit with required components
find_package(CUDAToolkit REQUIRED)

# Find third-party dependencies
find_package(Eigen3 REQUIRED)    # Linear algebra library
find_package(GLEW REQUIRED)      # OpenGL Extension Wrangler
find_package(GLUT REQUIRED)      # OpenGL Utility Toolkit
find_package(OpenGL REQUIRED)    # Open Graphics Library

# =============================================================================
# Target Configuration
# =============================================================================

# Create main executable target
add_executable(gmpm)

# Set CUDA-specific target properties
set_target_properties(gmpm PROPERTIES
    CUDA_EXTENSIONS OFF                    # Disable CUDA compiler extensions
    CUDA_SEPARABLE_COMPILATION ON          # Enable separate compilation
    CUDA_RESOLVE_DEVICE_SYMBOLS ON         # Resolve device symbols at link time
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} # Target CUDA architecture
)

# =============================================================================
# Source File Management
# =============================================================================

# Recursively collect all CUDA and C++ source files from GPU_MPM directory
file(GLOB_RECURSE gmpm_SOURCE CONFIGURE_DEPENDS 
    "GPU_MPM/*.cu"     # CUDA source files
    "GPU_MPM/*.cpp"    # C++ source files
)

# Recursively collect all header files from GPU_MPM directory
file(GLOB_RECURSE gmpm_HEADER CONFIGURE_DEPENDS 
    "GPU_MPM/*.h"      # C header files
    "GPU_MPM/*.cuh"    # CUDA header files
    "GPU_MPM/*.hpp"    # C++ header files
    "GPU_MPM/*.inl"    # Inline files
    "GPU_MPM/*.inc"    # Include files
)

# Add all collected sources to the target
target_sources(gmpm PRIVATE ${gmpm_SOURCE} ${gmpm_HEADER})

# =============================================================================
# Include Directories
# =============================================================================

# Set include directories for the target
target_include_directories(gmpm PRIVATE 
    "GPU_MPM/"                 # Main source directory
    ${Eigen3_INCLUDE_DIRS}     # Eigen3 include directories
)

# =============================================================================
# Compile Definitions
# =============================================================================

# Set preprocessor definitions
target_compile_definitions(gmpm PRIVATE 
    gmpm_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/"    # Assets directory path
    gmpm_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/"    # Output directory path
)

# =============================================================================
# Library Linking
# =============================================================================

# Link CUDA libraries
target_link_libraries(gmpm PRIVATE
    CUDA::cudart      # CUDA runtime library
    CUDA::cublas      # CUDA BLAS library
    CUDA::cusolver    # CUDA solver library
    CUDA::cusparse    # CUDA sparse matrix library
    CUDA::cufft       # CUDA FFT library
)

# Link graphics and math libraries
target_link_libraries(gmpm PRIVATE
    GLUT::GLUT        # GLUT library
    GLEW::GLEW        # GLEW library
    OpenGL::GLU    # OpenGL library
    Eigen3::Eigen     # Eigen3 library
)

# =============================================================================
# Compiler Options
# =============================================================================

# Set CUDA-specific compiler options
target_compile_options(gmpm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:  # Apply only to CUDA files
        --use_fast_math          # Use fast math operations
        -lineinfo                # Generate line information for profiling
        --ptxas-options=-allow-expensive-optimizations=true,-v  # PTX assembler options
        --extended-lambda        # Enable extended lambda support
        --expt-relaxed-constexpr # Enable relaxed constexpr support
        --default-stream=per-thread  # Use per-thread default stream
    >
)

# =============================================================================
# IDE Support and Organization
# =============================================================================

# Organize source files in IDE projects
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${gmpm_SOURCE})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${gmpm_HEADER})

# =============================================================================
# Configuration Summary
# =============================================================================

# Display configuration information
message(STATUS "Project: GPU_MPM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CUDA_ARCHITECTURES}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Source files found: ${gmpm_SOURCE}")
message(STATUS "Header files found: ${gmpm_HEADER}")