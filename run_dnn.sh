#!/bin/bash

source ../.bashrc

mkdir -p build
cd build || { echo "Failed to change directory to 'build'. Aborting..."; exit 1; }

rm -r *

cmake .. -DCMAKE_CUDA_ARCHITECTURES=native -DPHANTOM_ENABLE_DNN=ON

cd ..
cmake --build build -j 

./build/bin/DNN 

