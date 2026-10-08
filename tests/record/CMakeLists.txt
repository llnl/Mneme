# The first argument is the executable to generate, the second arguments is the
# source file with checks. Any following arguments are other source files
# (without checks) to compile for generating the executable.
function(CREATE_GPU_TEST exe check_source)
  add_executable(${exe} ${check_source} ${ARGN})
  if (MNEME_ENABLE_HIP)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE HIP) 
    target_compile_definitions(${exe} PRIVATE MNEME_ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
  elseif(MNEME_ENABLE_CUDA)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE CUDA) 
    target_compile_definitions(${exe} PRIVATE MNEME_ENABLE_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart CUDA::cuda_driver)
  endif()
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/include/)

  add_dependencies(${exe} record)
  add_mneme(${exe})

add_test(NAME ${exe} COMMAND ${LIT} -a -vv -DFILECHECK=${FILECHECK} -D ext='' -D PG=1 -D RR=${RECORD_READ_JSON} ${check_source})
  set_tests_properties(${exe} PROPERTIES LABELS "MnemeRecord::Link")
endfunction()

function(CREATE_RDC_GPU_TEST exe check_source)
  add_executable(${exe}.rdc ${check_source} ${ARGN})
  if (MNEME_ENABLE_HIP)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE HIP) 
    target_compile_definitions(${exe}.rdc PRIVATE MNEME_ENABLE_HIP)
    target_link_libraries(${exe}.rdc PRIVATE hip::host)
  elseif(MNEME_ENABLE_CUDA)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE CUDA) 
    target_compile_definitions(${exe}.rdc PRIVATE MNEME_ENABLE_CUDA)
    target_link_libraries(${exe}.rdc PRIVATE CUDA::cudart CUDA::cuda_driver)
  endif()
  target_include_directories(${exe}.rdc PRIVATE ${CMAKE_SOURCE_DIR}/include/)

  add_dependencies(${exe}.rdc record)

  if(MNEME_ENABLE_HIP)
    # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

    # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
    target_compile_options(
        ${exe}.rdc
        PUBLIC
        -fgpu-rdc
    )

    target_link_options(
        ${exe}.rdc
        PUBLIC
        -fgpu-rdc
        --hip-link
    )
    elseif(MNEME_ENABLE_CUDA)
        target_compile_options(
            ${exe}.rdc
            PUBLIC
            --no-offload-new-driver
        )
    set_target_properties(${exe}.rdc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
  add_mneme(${exe}.rdc)

  add_test(NAME ${exe}.rdc COMMAND ${LIT} -a -vv -DFILECHECK=${FILECHECK} -D ext=.rdc -D PG=1 -D RR=${RECORD_READ_JSON} ${check_source})
  set_tests_properties(${exe}.rdc PROPERTIES LABELS "MnemeRecord::RDC")
endfunction()


find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(RECORD_READ_JSON "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/read-record.py")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/read-record.py ${CMAKE_CURRENT_BINARY_DIR}/read-record.py COPYONLY)



file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats
import os
import tempfile
import atexit
import shutil

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
# Create a unique temp exec_root per ctest invocation to avoid races on
# lit_test_times.txt and on test artifacts under ctest -jN.
exec_root = tempfile.mkdtemp(prefix='lit.tmp.', dir='${CMAKE_CURRENT_BINARY_DIR}')
config.test_exec_root = exec_root
atexit.register(lambda: shutil.rmtree(exec_root, ignore_errors=False))

MNEME_PRELOAD_LIB='${CMAKE_BINARY_DIR}/src/librecord.so'
PAGE_SIZE = lit_config.params['PG']
RR = lit_config.params['RR']
EXT = lit_config.params['ext']
FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('MNEME_PRELOAD_LIB', MNEME_PRELOAD_LIB))
config.substitutions.append(('%PG', PAGE_SIZE))
config.substitutions.append(('%RR', RR))
config.substitutions.append(('%ext', EXT))
config.substitutions.append(('%FILECHECK', FILECHECK))
config.substitutions.append(('%build', '${CMAKE_CURRENT_BINARY_DIR}'))
"
)

CREATE_GPU_TEST(kernel kernel.cpp)
CREATE_RDC_GPU_TEST(kernel kernel.cpp)

CREATE_GPU_TEST(test_block_grid_1d test_block_grid_1d.cpp)
CREATE_RDC_GPU_TEST(test_block_grid_1d test_block_grid_1d.cpp)

CREATE_GPU_TEST(test_block_grid_2d test_block_grid_2d.cpp)
CREATE_RDC_GPU_TEST(test_block_grid_2d test_block_grid_2d.cpp)

CREATE_GPU_TEST(test_block_grid_3d test_block_grid_3d.cpp)
CREATE_RDC_GPU_TEST(test_block_grid_3d test_block_grid_3d.cpp)

CREATE_GPU_TEST(test_skip_recordings test_skip_recordings.cpp)
CREATE_RDC_GPU_TEST(test_skip_recordings test_skip_recordings.cpp)

CREATE_GPU_TEST(test_launch_counts test_launch_counts.cpp)
CREATE_RDC_GPU_TEST(test_launch_counts test_launch_counts.cpp)

CREATE_GPU_TEST(test_multi_file multi_file1.cpp multi_file2.cpp)
CREATE_RDC_GPU_TEST(test_multi_file multi_file1.cpp multi_file2.cpp)

CREATE_GPU_TEST(test_multi_file_launcher multi_file1_kernel_launcher.cpp multi_file2_kernel_launcher.cpp)
CREATE_RDC_GPU_TEST(test_multi_file_launcher multi_file1_kernel_launcher.cpp multi_file2_kernel_launcher.cpp)

CREATE_GPU_TEST(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_RDC_GPU_TEST(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)

CREATE_GPU_TEST(multi_kernel_regex multi_kernel_regex.cpp)
CREATE_RDC_GPU_TEST(multi_kernel_regex multi_kernel_regex.cpp)

CREATE_GPU_TEST(test_annotation test_annotation.cpp)
CREATE_RDC_GPU_TEST(test_annotation test_annotation.cpp)

CREATE_GPU_TEST(test_crash_handling test_crash_handling.cpp)

CREATE_GPU_TEST(test_host_access test_host_access.cpp)

CREATE_GPU_TEST(test_source_file test_source_file.cpp)
CREATE_RDC_GPU_TEST(test_source_file test_source_file.cpp)

CREATE_GPU_TEST(test_source_file_no_debug test_source_file_no_debug.cpp)
# Disable debug info to check that the missing-line-table warning prints.
target_compile_options(test_source_file_no_debug PRIVATE -g0)
