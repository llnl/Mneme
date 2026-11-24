cmake_minimum_required(VERSION 3.21)
project(mneme_fidelity LANGUAGES C CXX)

find_package(mneme REQUIRED CONFIG)
set(FIDELITY_BACKEND "hip" CACHE STRING "hip or cuda")

if(FIDELITY_BACKEND STREQUAL "hip")
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
elseif(FIDELITY_BACKEND STREQUAL "cuda")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
else()
  message(FATAL_ERROR "FIDELITY_BACKEND must be hip or cuda")
endif()

# Keep optimization and debug information identical in both executables.
# Debug metadata is intentional: it exercises the normal replay compilation path.
foreach(target fidelity_native fidelity_record)
  add_executable(${target} main.cpp)
  target_compile_options(${target} PRIVATE
    -O3
    -gline-tables-only
    -fno-discard-value-names
    -ftrivial-auto-var-init=zero
  )
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    HIP_STANDARD 17
    CUDA_STANDARD 17
  )
  target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})

  if(FIDELITY_BACKEND STREQUAL "hip")
    target_compile_definitions(${target} PRIVATE FIDELITY_HIP)
    target_link_libraries(${target} PRIVATE hip::host)
  else()
    target_link_libraries(${target} PRIVATE CUDA::cudart)
  endif()
endforeach()

# Recording instrumentation is the only build difference.
add_mneme(fidelity_record)
