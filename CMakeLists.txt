cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(accessor_example LANGUAGES CXX)

option(ACCESSOR_CUDA "Build the CUDA benchmark" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(GINKGO_DIR "" CACHE PATH
    "Directory of Ginkgo")

string(COMPARE EQUAL "${GINKGO_DIR}" "" empty_ginkgo_dir)
# If directory not specified, try to find it. If not found, clone it locally
if (empty_ginkgo_dir)
    #find_package("Ginkgo")
    #if (Ginkgo_FOUND)
    #    message(STATUS "Using external version of package Ginkgo. In case of problems, consider setting the directory of Ginkgo manually with -DGINKGO_DIR=<directory/to/Ginkgo>.")
    #else()
    #endif()
    message(STATUS "GINKGO_DIR was not specified. Cloning Ginkgo into build directory...")
    # TODO
endif()

function(example_apply_default_target_settings target)
    target_compile_features("${target}" PUBLIC cxx_std_14)
    target_include_directories("${target}" PRIVATE
        "${GINKGO_DIR}"
        )
endfunction()


function(create_cuda_executable target cublas target_source)
    #message(STATUS "Parameter: ${target} ${cublas} ${target_source}")
    add_executable("${target}")
    example_apply_default_target_settings("${target}")
    # Force the language to be CUDA (works, but the *.cu approach is cleaner)
    # set_source_files_properties(main.cpp.inc PROPERTIES LANGUAGE CUDA)
    
    target_sources("${target}" PRIVATE
        ${target_source}
        )

    #TODO maybe add the Ginkgo Architecture Selector in this project
    #     OR simply use `CUDA_ARCHITECTURES` to choose the fitting architecture
    target_compile_options("${target}" PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        # Need atomicAdd for double -> compute_60
        $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_60>
        $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_70,sm_72,sm_75>
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>
        #$<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_80,code=sm_80>
        # Make sure the cpp files are treated as cu files
        #$<$<COMPILE_LANGUAGE:CUDA>:-x cu>
        )
    target_link_libraries("${target}" PRIVATE "${cublas}")

    #target_include_directories("${target}" PRIVATE
    #    "${GINKGO_DIR}"
    # Include CUDA header directory in cpp files
    #"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

endfunction()

if (ACCESSOR_CUDA)
    enable_language(CUDA)
    find_library(CUBLAS cublas
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH)

    create_cuda_executable("cuda_gemv" "${CUBLAS}" "cuda/gemv_benchmark.cu")
    create_cuda_executable("cuda_dot" "${CUBLAS}" "cuda/dot_benchmark.cu")


endif()
