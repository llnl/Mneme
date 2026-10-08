function(BUILD_DEVICE_UNIT_TEST exe)
  add_executable(${exe} ${CMAKE_SOURCE_DIR}/src/MnemeDeviceCode.cpp  ${CMAKE_SOURCE_DIR}/src/MnemeAnnotationInternal.cpp ${CMAKE_SOURCE_DIR}/src/MnemeAnnotation.cpp ${ARGN})
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/ ${CMAKE_SOURCE_DIR}/include/ ${LLVM_INCLUDE_DIRS})
  if (MNEME_ENABLE_HIP)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(${exe} PRIVATE hip::host proteus)
  elseif(MNEME_ENABLE_CUDA)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/MnemeDeviceCode.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart CUDA::cuda_driver proteus)
  endif()

  if (LLVM_LINK_LLVM_DYLIB)
	llvm_config(${exe} USE_SHARED)
	target_link_libraries(${exe} PRIVATE ${CMAKE_DL_LIBS})
   else()
	target_link_libraries(${exe} PRIVATE ${MNEME_LLVM_LIBS} ${CMAKE_DL_LIBS})
   endif()


if (MNEME_ENABLE_LOGGER)
target_link_libraries(${exe} PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(${exe} PRIVATE MNEME_ENABLE_LOGGER)
endif()

endfunction()


function(BUILD_UNIT_TEST exe)
  message(STATUS "Building test ${exe} ${ARGN}")
  add_executable(${exe} ${CMAKE_SOURCE_DIR}/src/MnemeAnnotationInternal.cpp ${CMAKE_SOURCE_DIR}/src/MnemeAnnotation.cpp ${ARGN})
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/ ${CMAKE_SOURCE_DIR}/include/ ${LLVM_INCLUDE_DIRS})
  if (MNEME_ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host proteus)
  elseif(MNEME_ENABLE_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart CUDA::cuda_driver proteus)
  endif()
  if (LLVM_LINK_LLVM_DYLIB)
	llvm_config(${exe} USE_SHARED)
	target_link_libraries(${exe} PRIVATE ${CMAKE_DL_LIBS})
   else()
	target_link_libraries(${exe} PRIVATE ${MNEME_LLVM_LIBS} ${CMAKE_DL_LIBS})
   endif()

if (MNEME_ENABLE_LOGGER)
target_link_libraries(${exe} PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(${exe} PRIVATE MNEME_ENABLE_LOGGER)
endif()

endfunction()

BUILD_UNIT_TEST("Config" Config.cpp)
add_test(NAME Config::Runtime COMMAND Config)

BUILD_UNIT_TEST("SerializeGlobals" SerializeGlobals.cpp)
add_test(NAME Serialize::Globals COMMAND SerializeGlobals)

BUILD_UNIT_TEST("SerializeBlob" SerializeBlob.cpp)
add_test(NAME Serialize::Blob COMMAND SerializeBlob)

BUILD_UNIT_TEST("ReadWriteSnapshot" ReadWriteSnapshot.cpp)
add_test(NAME Serialize::RWSnapshot COMMAND ReadWriteSnapshot)

BUILD_DEVICE_UNIT_TEST("CompareMemoryBlob" DeviceKernels.cpp)
add_test(NAME DeviceKernel::CompareMemory COMMAND CompareMemoryBlob)

BUILD_UNIT_TEST("VASpace" VASpace.cpp)
add_test(NAME VASpace::FirstFree COMMAND VASpace)
