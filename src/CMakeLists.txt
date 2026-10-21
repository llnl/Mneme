
set(MnemeCoreSrcDir ${CMAKE_CURRENT_SOURCE_DIR})
set(AnnotationRuntimeSRC MnemeAnnotation.cpp MnemeRecordScope.cpp)
set(RecordLibSRC MnemeAnnotationInternal.cpp MnemeCrashHandler.cpp MnemeRecordingPolicy.cpp MnemeKernelTimer.cpp)

if (MNEME_ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordPreloadHIP.cpp)
elseif(MNEME_ENABLE_CUDA)
  list(APPEND RecordLibSRC MnemeRecordPreloadCUDA.cpp)
endif()

# Vendor profiler used to time kernels, by recording and by replay profiling.
if (MNEME_ENABLE_HIP)
  find_package(rocprofiler-sdk QUIET CONFIG)
  if(rocprofiler-sdk_FOUND AND TARGET rocprofiler-sdk::rocprofiler-sdk)
    set(MNEME_PROFILER_LIBS rocprofiler-sdk::rocprofiler-sdk)
  else()
    message(WARNING "CMake package for rocprofiler-sdk not found; falling back to manual include/lib lookup")
    find_path(ROCPROFILER_SDK_INCLUDE_DIR rocprofiler-sdk/rocprofiler.h
            HINTS "${ROCM_PATH}/include")
    find_library(ROCPROFILER_SDK_LIBRARY NAMES rocprofiler-sdk
               HINTS "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64")
    if(NOT ROCPROFILER_SDK_INCLUDE_DIR OR NOT ROCPROFILER_SDK_LIBRARY)
      message(FATAL_ERROR "Could not find rocprofiler-sdk headers or library")
    endif()
    set(MNEME_PROFILER_INCLUDE_DIRS "${ROCPROFILER_SDK_INCLUDE_DIR}")
    set(MNEME_PROFILER_LIBS "${ROCPROFILER_SDK_LIBRARY}")
  endif()
elseif(MNEME_ENABLE_CUDA)
  set(MNEME_PROFILER_LIBS CUDA::cupti)
endif()

add_library(mnemert SHARED ${AnnotationRuntimeSRC})
target_include_directories(
 mnemert
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${_mneme_incdir}>)

add_library(record SHARED ${RecordLibSRC})

if(MNEME_LINK_SHARED_LLVM AND LLVM_LINK_LLVM_DYLIB)
  llvm_config(record USE_SHARED)
endif()

if (MNEME_ENABLE_HIP)
  set_source_files_properties(MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
elseif(MNEME_ENABLE_CUDA)
  set_source_files_properties(MnemeDeviceCode.cpp PROPERTIES LANGUAGE CUDA)
endif()


target_include_directories(record SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_include_directories(record PRIVATE ${MNEME_PROFILER_INCLUDE_DIRS})
target_link_libraries(record PRIVATE ${MNEME_PROFILER_LIBS})
target_link_libraries(record PRIVATE mnemert)
# Link record against proteus to avoid LLVM static initialization issues.
# HIP: proteus is shared (version-script hides LLVM symbols).
# CUDA: proteus is static but propagates --exclude-libs for the same effect.
target_link_libraries(record PRIVATE proteus)

if (MNEME_ENABLE_LOGGER)
  message(STATUS "Adding SPDLOG in recording")
  target_link_libraries(mnemert PRIVATE spdlog::spdlog_header_only)
  target_compile_definitions(mnemert PRIVATE MNEME_ENABLE_LOGGER)
  target_link_libraries(record PRIVATE spdlog::spdlog_header_only)
  target_compile_definitions(record PRIVATE MNEME_ENABLE_LOGGER)
endif()

if(MNEME_ENABLE_HIP)
  target_link_libraries(record PRIVATE hip::host hsa-runtime64::hsa-runtime64)
  target_link_libraries(record PRIVATE ${MNEME_LLVM_LIBS})
elseif(MNEME_ENABLE_CUDA)
  target_link_libraries(record PRIVATE CUDA::cudart CUDA::cuda_driver)
  target_link_libraries(record PRIVATE ${MNEME_LLVM_LIBS})
endif()

set_target_properties(record PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(pass)

if (MNEME_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
