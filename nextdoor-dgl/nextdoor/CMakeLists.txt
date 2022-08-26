cmake_minimum_required(VERSION 3.15.0)
project(nextdoor LANGUAGES CUDA CXX)

set(CMAKE_PREFIX_PATH "/home/ubuntu/anaconda3/envs/dgl/lib/python3.9/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("src")
include_directories("AnyOption")
# Define our library target
add_library(nextdoor SHARED
    src/khop/pybind.cu
    AnyOption/anyoption.cpp)

set_target_properties(nextdoor PROPERTIES CUDA_ARCHITECTURES "86")
target_compile_features(nextdoor PRIVATE cxx_std_14)
target_link_libraries(nextdoor ${TORCH_LIBRARIES} ${CUDA_cusparse_LIBRARY} )
