add_subdirectory(profile)

file(GLOB LLVM_SRC "llvm/*.cpp")
file(GLOB PROTEUS_SRC "proteus/*.cpp")
file(GLOB MNEME_SRC "./*.cpp")


add_library(mneme SHARED ${LLVM_SRC} ${PROTEUS_SRC} ${MNEME_SRC} ${MnemeCoreSrcDir}/MnemePageManager.cpp ${MnemeCoreSrcDir}/MnemeDeviceCode.cpp ${MnemeCoreSrcDir}/MnemeAnnotationInternal.cpp)
if (MNEME_ENABLE_HIP)
  set_source_files_properties(${MnemeCoreSrcDir}/MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(mneme PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(mneme PRIVATE MNEME_ENABLE_HIP)
elseif(MNEME_ENABLE_CUDA)
  set_source_files_properties(${MnemeCoreSrcDir}/MnemeDeviceCode.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(mneme PRIVATE MNEME_ENABLE_CUDA)
endif()

if(MNEME_LINK_SHARED_LLVM AND LLVM_LINK_LLVM_DYLIB)
  llvm_config(mneme USE_SHARED)
endif()
target_link_libraries(mneme PRIVATE ${MNEME_LLVM_LIBS})


target_include_directories(mneme SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
# FIXME: Here now we need to be "careful". If LLVM brings in the bundled .so we need to 
# link in proteusCore and be done with it.
target_link_libraries(mneme PRIVATE proteusCore)

if (MNEME_ENABLE_LOGGER)
  target_link_libraries(mneme PRIVATE spdlog::spdlog_header_only)
  target_compile_definitions(mneme PRIVATE MNEME_ENABLE_LOGGER)
endif()

if(MNEME_ENABLE_HIP)
  # Proteus's HIP impl headers (CoreLLVMHIP.h) call lld::elf::link inline,
  # so consumers that instantiate that code must link the LLD libraries
  # themselves — libproteus.so hides them via its version script.
  target_link_libraries(mneme PRIVATE hip::host hsa-runtime64::hsa-runtime64 lldCommon lldELF)
elseif(MNEME_ENABLE_CUDA)
  target_link_libraries(mneme PRIVATE CUDA::cudart CUDA::cuda_driver)
  if(MNEME_NVPTXCOMPILER_LIBRARY MATCHES "\\.a$")
    get_filename_component(_mneme_nvptxcompiler_name "${MNEME_NVPTXCOMPILER_LIBRARY}" NAME)
    target_link_libraries(mneme PRIVATE
      -Wl,--whole-archive
      "${MNEME_NVPTXCOMPILER_LIBRARY}"
      -Wl,--no-whole-archive
      "-Wl,--exclude-libs,${_mneme_nvptxcompiler_name}")
  else()
    target_link_libraries(mneme PRIVATE "${MNEME_NVPTXCOMPILER_LIBRARY}")
  endif()
endif()


set_target_properties(mneme PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
