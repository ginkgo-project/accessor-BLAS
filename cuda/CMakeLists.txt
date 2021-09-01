# After `target_source`, list libraries to link against
function(create_cuda_executable target target_source)
    add_executable("${target}")
    example_apply_default_target_settings("${target}")
    
    target_sources("${target}" PRIVATE
        ${target_source}
        )

    # The CUDA architecture should be chosen with the `CMAKE_CUDA_ARCHITECTURES`,
    # which is built into CMake
    target_compile_options("${target}" PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    target_link_libraries("${target}" PRIVATE "${ARGN}")

endfunction()

enable_language(CUDA)
find_library(CUBLAS cublas
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH)
find_library(CUSOLVER cusolver
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH)

create_cuda_executable("gemv_benchmark" "gemv_benchmark.cu" "${CUBLAS}")
create_cuda_executable("dot_benchmark" "dot_benchmark.cu" "${CUBLAS}")
create_cuda_executable("trsv_benchmark" "trsv_benchmark.cu" "${CUBLAS}" "${CUSOLVER}")
create_cuda_executable("ir_benchmark" "ir.cu" "${CUBLAS}" "${CUSOLVER}")
