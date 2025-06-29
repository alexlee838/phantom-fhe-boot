add_executable(DNN
dnn_example.cu)
target_link_libraries(DNN PRIVATE Phantom)
target_include_directories(DNN PUBLIC ${CMAKE_SOURCE_DIR}/include)
