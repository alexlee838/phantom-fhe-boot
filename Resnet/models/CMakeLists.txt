# Resnet/models/CMakeLists.txt

# You can glob or manually list the kernel sources
file(GLOB MODEL_KERNELS "*.cu")

set(MODELS_SRCS ${MODEL_KERNELS} PARENT_SCOPE)
