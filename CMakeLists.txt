cmake_minimum_required(VERSION 3.20)

project(Phantom LANGUAGES CXX CUDA VERSION 2.1 DESCRIPTION "CUDA-Accelerated Homomorphic Encryption Library")

# [option] PHANTOM_USE_CUDA_PTX (default: ON)
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX Assembly" ON)
message(STATUS "Use CUDA PTX Assembly: ${PHANTOM_USE_CUDA_PTX}")
if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif ()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

# use C++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# library
add_subdirectory(src)

# example
# [option] PHANTOM_ENABLE_EXAMPLE (default: ON)
option(PHANTOM_ENABLE_EXAMPLE "Enable examples" ON)
message(STATUS "Enable examples: ${PHANTOM_ENABLE_EXAMPLE}")
if (PHANTOM_ENABLE_EXAMPLE)
    add_subdirectory(examples)
endif ()

# benchmark
# [option] PHANTOM_ENABLE_BENCH (default: OFF)
option(PHANTOM_ENABLE_BENCH "Enable benchmarks" OFF)
message(STATUS "Enable benchmarks: ${PHANTOM_ENABLE_BENCH}")
if (PHANTOM_ENABLE_BENCH)
    add_subdirectory(benchmark)
endif ()

# test
# [option] PHANTOM_ENABLE_TEST (default: OFF)
option(PHANTOM_ENABLE_TEST "Enable tests" OFF)
message(STATUS "Enable tests: ${PHANTOM_ENABLE_TEST}")
if (PHANTOM_ENABLE_TEST)
    add_subdirectory(test)
endif ()

# python binding
# [option] PHANTOM_ENABLE_PYTHON_BINDING (default: OFF)
option(PHANTOM_ENABLE_PYTHON_BINDING "Enable python binding" OFF)
message(STATUS "Enable python binding: ${PHANTOM_ENABLE_PYTHON_BINDING}")
if (PHANTOM_ENABLE_PYTHON_BINDING)
    add_subdirectory(python)
endif ()

# BootStrap
# [option] PHANTOM_ENABLE_BOOTSTRAP (default: ON)
option(PHANTOM_ENABLE_BOOTSTRAP "BootStrap" ON)
message(STATUS "Enable BootStrapping: ${PHANTOM_ENABLE_BOOTSTRAP}")
if (PHANTOM_ENABLE_BOOTSTRAP)
    add_subdirectory(bootstrapping)
endif ()

# Convolution
# [option] PHANTOM_ENABLE_Convolution (default: OFF)
option(PHANTOM_ENABLE_CONVOLUTION "Convolution" OFF)
message(STATUS "Enable Convolution: ${PHANTOM_ENABLE_CONVOLUTION}")
if (PHANTOM_ENABLE_CONVOLUTION)
    add_subdirectory(convolution)
endif ()

# DNN
# [option] PHANTOM_ENABLE_DNN (default: OFF)
option(PHANTOM_ENABLE_DNN "DNN" OFF)
message(STATUS "Enable DNN: ${PHANTOM_ENABLE_DNN}")
if (PHANTOM_ENABLE_DNN)
    add_subdirectory(DNN)
endif ()

# ResNet
# [option] PHANTOM_ENABLE_DNN (default: OFF)
option(PHANTOM_ENABLE_RESNET "RESNET" OFF)
message(STATUS "Enable RESNET: ${PHANTOM_ENABLE_RESNET}")
if (PHANTOM_ENABLE_RESNET)
    add_subdirectory(Resnet)
endif ()

# config for installation
install(TARGETS Phantom EXPORT PhantomConfig)
install(EXPORT PhantomConfig NAMESPACE phantom:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phantom)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/phantom)

# Attempt to locate and enable OpenMP
find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found! Enabling OpenMP support.")
else()
    message(WARNING "OpenMP not found. Compilation will proceed without OpenMP support.")
endif()

if (OpenMP_CXX_FOUND)
    # Assuming "Phantom" is the target name
    target_link_libraries(Phantom PUBLIC OpenMP::OpenMP_CXX)
endif()
