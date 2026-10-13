if (MNEME_ENABLE_HIP)
  add_library(mneme_profile SHARED rocr_profile.cpp)

  target_include_directories(mneme_profile PRIVATE ${HIP_INCLUDE_DIRS})
  target_link_libraries(mneme_profile PRIVATE hip::host)
elseif(MNEME_ENABLE_CUDA)
  add_library(mneme_profile SHARED cupti_profile.cpp)
  target_link_libraries(mneme_profile PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

target_include_directories(mneme_profile PRIVATE ${MNEME_PROFILER_INCLUDE_DIRS})
target_link_libraries(mneme_profile PRIVATE ${MNEME_PROFILER_LIBS} dl pthread)

if (MNEME_ENABLE_LOGGER)
  target_link_libraries(mneme_profile PRIVATE spdlog::spdlog_header_only)
  target_compile_definitions(mneme_profile PRIVATE MNEME_ENABLE_LOGGER)
endif()
target_include_directories(mneme_profile PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(mneme::profile ALIAS mneme_profile)

set_target_properties(mneme_profile PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
