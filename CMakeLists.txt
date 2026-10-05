cmake_minimum_required(VERSION 3.28.2)

# compiler auto-selection from LLVM_INSTALL_DIR, must happen before project()
# this will override compiler selection from tools like Spack, which inject compiler wrappers
# that will make mneme config c/cxx unusable
if(NOT LLVM_INSTALL_DIR)
  set(LLVM_INSTALL_DIR "$ENV{LLVM_INSTALL_DIR}" CACHE PATH
    "Path to the LLVM installation used to build Mneme")
endif()

set(MNEME_GPU_BACKEND "$ENV{MNEME_GPU_BACKEND}" CACHE STRING "GPU backend (cuda or hip)")
set_property(CACHE MNEME_GPU_BACKEND PROPERTY STRINGS cuda hip)
string(TOLOWER "${MNEME_GPU_BACKEND}" MNEME_GPU_BACKEND)
if(MNEME_GPU_BACKEND STREQUAL "cuda")
  set(MNEME_ENABLE_CUDA ON)
elseif(MNEME_GPU_BACKEND STREQUAL "hip")
  set(MNEME_ENABLE_HIP ON)
else()
  message(FATAL_ERROR
    "MNEME_GPU_BACKEND must be \"cuda\" or \"hip\" (got \"${MNEME_GPU_BACKEND}\").\n"
    "Set it as an environment variable or pass -DMNEME_GPU_BACKEND")
endif()

if(NOT LLVM_INSTALL_DIR)
  message(FATAL_ERROR
    "LLVM_INSTALL_DIR is required.\n"
    "Set it as an environment variable or pass -DLLVM_INSTALL_DIR")
elseif(NOT EXISTS "${LLVM_INSTALL_DIR}")
  message(FATAL_ERROR "LLVM_INSTALL_DIR does not exist: ${LLVM_INSTALL_DIR}")
else()
  find_program(CMAKE_CXX_COMPILER
    NAMES amdclang++ clang++
    HINTS "${LLVM_INSTALL_DIR}/bin"
    NO_DEFAULT_PATH
  )
  find_program(CMAKE_C_COMPILER
    NAMES amdclang clang
    HINTS "${LLVM_INSTALL_DIR}/bin"
    NO_DEFAULT_PATH
  )
  # we want clang to be the cuda compiler
  if(MNEME_ENABLE_CUDA)
    find_program(CMAKE_CUDA_COMPILER
      NAMES clang++
      HINTS "${LLVM_INSTALL_DIR}/bin"
      NO_DEFAULT_PATH
    )
  endif()
  if(CMAKE_CXX_COMPILER)
    message(STATUS "Auto-selected CXX: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Auto-selected C:   ${CMAKE_C_COMPILER}")
    if(CMAKE_CUDA_COMPILER)
      message(STATUS "Auto-selected CUDA: ${CMAKE_CUDA_COMPILER}")
    endif()
  else()
    message(WARNING "Compiler auto-select: no amdclang++/clang++ found in ${LLVM_INSTALL_DIR}/bin")
  endif()
endif()

project(
  mneme
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MNEME_ENABLE_TESTS "Enable Testing" OFF)
option(MNEME_ENABLE_LOGGER "Enable logging on mneme" OFF)
option(MNEME_ENABLE_PYTHON "Build Python bindings, profiling, and autotuning" OFF)
option(MNEME_PYTHON_WHEEL "Build using Python wheel layout" OFF)
set(MNEME_PYTHON_SITE_PACKAGES "" CACHE PATH "Destination for Python files (non-wheel installs like Spack)")
# AUTO picks up whatever the found LLVM was built with
set(MNEME_LINK_SHARED_LLVM "AUTO" CACHE STRING "Link with the LLVM shared library")


if(MNEME_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # cmake's native arch detection doesn't work with Clang as the CUDA compiler until 3.28.2
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for")
  endif()
  enable_language(CUDA)
  find_library(MNEME_NVPTXCOMPILER_LIBRARY
    NAMES nvptxcompiler_static libnvptxcompiler_static nvptxcompiler
    REQUIRED
    NO_DEFAULT_PATH
    PATHS "${CUDAToolkit_LIBRARY_DIR}"
  )
  message(STATUS "CUDA nvPTXCompiler library: ${MNEME_NVPTXCOMPILER_LIBRARY}")
  add_compile_definitions(MNEME_ENABLE_CUDA)
elseif(MNEME_ENABLE_HIP)
  find_package(HIP REQUIRED)
  find_package(hsa-runtime64 REQUIRED)
  enable_language(HIP)
  add_compile_definitions(MNEME_ENABLE_HIP)
endif()

if(MNEME_ENABLE_LOGGER)
  include(FetchContent)
  block()
    if(MNEME_PYTHON_WHEEL)
      # Installed spdlog targets may link shared fmt, which is not bundled in the wheel.
      set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
      set(SPDLOG_FMT_EXTERNAL OFF)
      set(SPDLOG_FMT_EXTERNAL_HO OFF)
      set(SPDLOG_USE_STD_FORMAT OFF)
    endif()
    FetchContent_Declare(spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG        v1.15.0
      GIT_SHALLOW    TRUE
      FIND_PACKAGE_ARGS
    )
    FetchContent_MakeAvailable(spdlog)
  endblock()
endif()

message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

include(cmake/SetupLLVM.cmake)
include(cmake/SetupProteus.cmake)
include(cmake/MnemeFunctions.cmake)

# needs to go before add_subdirectory
if(MNEME_PYTHON_WHEEL)
  if(proteus_SOURCE_DIR)
    # bundled proteus is installed alongside mneme
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
  else()
    # set real proteus libdir b/c INSTALL_RPATH_USE_LINK_PATH won't pick up external proteus
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$<TARGET_FILE_DIR:proteus>")
  endif()
endif()

# install destinations: wheel uses mneme/native/* so artifacts are in the Python package
# standard installs use GNUInstallDirs vars
# needs to go before add_subdirectory so src/CMakeLists.txt can use _mneme_incdir
# for mnemert's exported INSTALL_INTERFACE include directory
if(MNEME_PYTHON_WHEEL)
  set(_mneme_libdir   "mneme/native/lib64")
  set(_mneme_incdir   "mneme/native/include")
  set(_mneme_cmakedir "mneme/native/lib64/cmake/mneme")
else()
  set(_mneme_libdir   "${CMAKE_INSTALL_LIBDIR}")
  set(_mneme_incdir   "${CMAKE_INSTALL_INCLUDEDIR}")
  set(_mneme_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/mneme")
endif()

add_subdirectory(src)

if(MNEME_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "Installing ${PROJECT_NAME} of version ${PROJECT_VERSION}")

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/mnemeConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/mnemeConfig.cmake"
  INSTALL_DESTINATION "${_mneme_cmakedir}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/mnemeConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

set(_mneme_targets mnemert record)
if(MNEME_ENABLE_PYTHON)
  list(APPEND _mneme_targets mneme_profile mneme MnemePassPlugin)
endif()

install(TARGETS ${_mneme_targets}
  EXPORT mnemeTargets
  LIBRARY DESTINATION "${_mneme_libdir}"
  RUNTIME DESTINATION "${_mneme_libdir}"
  ARCHIVE DESTINATION "${_mneme_libdir}")
install(FILES "${PROJECT_SOURCE_DIR}/include/mneme/MnemeAnnotation.hpp"
  DESTINATION "${_mneme_incdir}/mneme")
install(EXPORT mnemeTargets
  FILE mnemeTargets.cmake
  NAMESPACE Mneme::
  DESTINATION "${_mneme_cmakedir}")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/mnemeConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/mnemeConfigVersion.cmake"
  "${PROJECT_SOURCE_DIR}/cmake/MnemeFunctions.cmake"
  DESTINATION "${_mneme_cmakedir}")

# python package installation
if(MNEME_ENABLE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  # we need to track the proteus installation for the cflags and ldflags config commands
  if(NOT PROTEUS_LIBDIR)
    # proteus is bundled in the same prefix as mneme
    if(MNEME_PYTHON_WHEEL AND proteus_SOURCE_DIR)
      set(PROTEUS_LIBDIR "@PREFIX@/lib64")
    else()
      get_target_property(_proteus_link_dirs proteus INTERFACE_LINK_DIRECTORIES)
      if(_proteus_link_dirs)
        list(GET _proteus_link_dirs 0 PROTEUS_LIBDIR)
      else()
        message(FATAL_ERROR
          "Cannot determine PROTEUS_LIBDIR from the proteus target.")
      endif()
    endif()
  endif()
  message(STATUS "Proteus library directory: ${PROTEUS_LIBDIR}")

  if(MNEME_PYTHON_WHEEL)
    # command.py will substitute proper prefix when wheel-installed
    set(PREFIX "@PREFIX@")
    set(MNEME_CONFIG_LIBDIR "@PREFIX@/lib64")
  else()
    set(PREFIX "${CMAKE_INSTALL_PREFIX}")
    set(MNEME_CONFIG_LIBDIR "${CMAKE_INSTALL_FULL_LIBDIR}")
  endif()

  configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/config.json.in"
    "${CMAKE_BINARY_DIR}/python/mneme/native/config.json"
    @ONLY)

  if(MNEME_PYTHON_WHEEL)
    install(FILES "${CMAKE_BINARY_DIR}/python/mneme/native/config.json"
      DESTINATION "mneme/native")
    if(proteus_SOURCE_DIR)
      # only bundle proteus when built from source
      install(TARGETS proteus ProteusPass ProteusPassOffload
        LIBRARY DESTINATION "${_mneme_libdir}"
        RUNTIME DESTINATION "${_mneme_libdir}")
    endif()
  elseif(MNEME_PYTHON_SITE_PACKAGES)
    # if a site-packages directory is passed (like from spack, install python files there)
    install(FILES "${CMAKE_BINARY_DIR}/python/mneme/native/config.json"
      DESTINATION "${MNEME_PYTHON_SITE_PACKAGES}/mneme/native")
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/python/mneme/"
      DESTINATION "${MNEME_PYTHON_SITE_PACKAGES}/mneme"
      PATTERN "__pycache__" EXCLUDE
      PATTERN "*.pyc" EXCLUDE)
    configure_file(
      "${PROJECT_SOURCE_DIR}/cmake/mneme_entrypoint.in"
      "${CMAKE_BINARY_DIR}/mneme"
      @ONLY)
    install(PROGRAMS "${CMAKE_BINARY_DIR}/mneme"
      DESTINATION "${CMAKE_INSTALL_BINDIR}")
  else()
    message(WARNING
      "MNEME_ENABLE_PYTHON=ON but MNEME_PYTHON_SITE_PACKAGES not set. "
      "Python files will not be installed."
    )
  endif()
endif()
