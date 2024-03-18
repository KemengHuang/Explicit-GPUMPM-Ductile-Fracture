cmake_minimum_required(VERSION 3.18)    # support relative path, cuda_std_17, archive manipulation
#set(CMAKE_CUDA_ARCHITECTURES compute_SM_86)
message("===============================================================")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
endif()
set(CMAKE_CXX_FLAGS "-O3")
# project meta
# ============
set(CMAKE_CUDA_COMPILER "/usr/local/cuda11.8/bin")
project(gpu_ipc LANGUAGES   CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(ZS_ENABLE_WRANGLER OFF)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENMP ON)
set(ZS_ENABLE_OPENVDB OFF)
set(ZS_ENABLE_PARTIO OFF)

add_compile_definitions(gipc PRIVATE USE_SNK SVD_DOUBLE ADAPTIVE_KAPPA)

# add_subdirectory(zsgipc)
#include_directories(mLBVH)
file(GLOB srcs CONFIGURE_DEPENDS GPU_IPC/mLBVH/*.cu GPU_IPC/mLBVH/*.cpp)
add_executable(gipc ${srcs})
target_compile_features(gipc PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(gipc PRIVATE CUDA::cuda_driver CUDA::cusparse CUDA::cusolver CUDA::cudart)
set_target_properties(gipc
    PROPERTIES  CUDA_EXTENSIONS ON
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                CUDA_ARCHITECTURES "60"
                )
target_compile_options(gipc 
  PRIVATE       $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true -Xcudafe "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642">
  PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr --default-stream=per-thread>
)

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
if (MSVC)
    message("win build")
    target_include_directories(gipc PRIVATE 
        GPU_IPC/mLBVH/GL_LIB/glew_64/include
        GPU_IPC/mLBVH/GL_LIB/freeglut_64/include
        GPU_IPC/mLBVH)
    target_link_directories(gipc PRIVATE 
        GPU_IPC/mLBVH/GL_LIB/glew_64/libs/release
        GPU_IPC/mLBVH/GL_LIB/freeglut_64/libs/release
    )
    target_link_libraries(gipc PRIVATE
        glew32.lib
        freeglut.lib
    )
elseif(UNIX)
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(GLEW REQUIRED)
    include_directories(GPU_IPC/mLBVH)
    target_link_libraries(gipc PRIVATE OpenGL::OpenGL GLUT::GLUT GLEW::GLEW OpenGL::GLU Eigen3::Eigen TBB::tbb)
endif(MSVC)
