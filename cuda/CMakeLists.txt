# After `target_source`, list libraries to link against
function(create_cuda_executable target target_source)
    #message(STATUS "Parameter: ${target} ${target_source} ${ARGN}")
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
        #$<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_70,sm_72,sm_75>
        # Make sure the cpp files are treated as cu files
        #$<$<COMPILE_LANGUAGE:CUDA>:-x cu>
        )
    target_link_libraries("${target}" PRIVATE "${ARGN}")

    #target_include_directories("${target}" PRIVATE
    #    "${GINKGO_DIR}"
    # Include CUDA header directory in cpp files
    #"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

endfunction()

enable_language(CUDA)
find_library(CUBLAS cublas
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH)
find_library(CUSOLVER cusolver
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH)

create_cuda_executable("gemv_benchmark" "gemv_benchmark.cu" "${CUBLAS}")
create_cuda_executable("dot_benchmark" "dot_benchmark.cu" "${CUBLAS}")
create_cuda_executable("trsv_benchmark" "trsv_benchmark.cu" "${CUBLAS}" "${CUSOLVER}")
