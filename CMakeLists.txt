cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(accessor_example LANGUAGES CXX)

option(ACCESSOR_CUDA "Build the CUDA benchmark" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(GINKGO_DIR "$ENV{HOME}/projects/ginkgo_github" CACHE PATH
    "Directory of Ginkgo")

function(example_apply_default_target_settings target)
    target_compile_features("${target}" PUBLIC cxx_std_14)
    target_include_directories("${target}" PRIVATE
        "${GINKGO_DIR}"
        )
endfunction()


if (ACCESSOR_CUDA)
    enable_language(CUDA)
    set(target "accessor_cuda")
    add_executable("${target}")
    example_apply_default_target_settings("${target}")
    target_compile_definitions("${target}" PRIVATE ACCESSOR_CUDA_CODE)

    # Force the language to be CUDA (works, but the main.cu approach is cleaner)
    # set_source_files_properties(main.cpp.inc PROPERTIES LANGUAGE CUDA)
    target_sources("${target}" PRIVATE
        cuda/main.cu
        )


    #TODO maybe add the Ginkgo Architecture Selector in this project
    target_compile_options("${target}" PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_52>
        $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_70,sm_72,sm_75>
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>
        #$<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_80,code=sm_80>
        # Make sure the cpp files are treated as cu files
        #$<$<COMPILE_LANGUAGE:CUDA>:-x cu>
        )

    target_include_directories("${target}" PRIVATE
        "${GINKGO_DIR}"
    # Include CUDA header directory in cpp files
    #"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        )

endif()
