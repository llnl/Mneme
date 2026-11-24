cmake_minimum_required(VERSION 3.8)

project(cuda-example  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
enable_language(CUDA)
find_package(CUDAToolkit)

find_package(mneme REQUIRED)

add_executable(vecAdd vec_add.cu)

get_cmake_property(_targets TARGETS)
message(STATUS "Available targets: ${_targets}")

add_mneme(vecAdd)

set_source_files_properties(vec_add.cu PROPERTIES LANGUAGE CUDA)
message(STATUS "Pass is ${RRPassPath}")

target_link_libraries(vecAdd PRIVATE CUDA::cudart CUDA::cuda_driver)
set_target_properties(vecAdd PROPERTIES CUDA_RUNTIME_LIBRARY Shared)