add_executable(bootstrap
bootstrapping_example.cu)
target_link_libraries(bootstrap PRIVATE Phantom)
target_include_directories(bootstrap PUBLIC ${CMAKE_SOURCE_DIR}/include)

