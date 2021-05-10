cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(accessor_example LANGUAGES CXX)

option(ACCESSOR_CUDA "Build the CUDA benchmark" ON)

set(GINKGO_DIR_INFO "Directory of Ginkgo (used for the accessor only)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(GINKGO_DIR "" CACHE PATH
    "${GINKGO_DIR_INFO}")

string(COMPARE EQUAL "${GINKGO_DIR}" "" empty_ginkgo_dir)
# If directory not specified, try to find it. If not found, clone it locally
if (empty_ginkgo_dir)
    message(STATUS "GINKGO_DIR was not specified. Cloning Ginkgo into build directory...")
    find_package(Git REQUIRED)

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --branch develop
                    "https://github.com/ginkgo-project/ginkgo.git" "Ginkgo"
        WORKING_DIRECTORY "${accessor_example_BINARY_DIR}"
        RESULT_VARIABLE result
        )
    if(result)
        message(FATAL_ERROR "Failed to clone Ginkgo repository. Error: ${result}")
    endif()
    set(GINKGO_DIR "${accessor_example_BINARY_DIR}/Ginkgo" CACHE PATH "${GINKGO_DIR_INFO}" FORCE)
    message(STATUS "Ginkgo successfully cloned into \"${GINKGO_DIR}\"")
endif()

function(example_apply_default_target_settings target)
    target_compile_features("${target}" PUBLIC cxx_std_14)
    target_include_directories("${target}" PRIVATE
        "${GINKGO_DIR}"
        )
endfunction()


if (ACCESSOR_CUDA)
    add_subdirectory(cuda)
endif()
