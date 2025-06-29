# Add the models subdirectory
add_subdirectory(models)

# Include cnpy headers
include_directories(${CMAKE_SOURCE_DIR}/external/cnpy)

# Include zlib headers (local install)
include_directories(/home/student/temp/shlee/.local/include)
link_directories(/home/student/temp/shlee/.local/lib)

# Define source files
set(RESNET_SRCS
    resnet.cu
    ${MODELS_SRCS}
    ${CMAKE_SOURCE_DIR}/external/cnpy/cnpy.cpp  # Add cnpy source
)

# Define the executable
add_executable(FHE_RESNET20 ${RESNET_SRCS})

# Link libraries
target_link_libraries(FHE_RESNET20
    PRIVATE
        Phantom
        z          # zlib from local install
)

# Include your main and models include paths
target_include_directories(FHE_RESNET20
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/models
)
