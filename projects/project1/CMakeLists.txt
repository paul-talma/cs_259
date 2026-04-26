cmake_minimum_required(VERSION 3.18)
project(cs_259 CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(conv project1/conv.cu)
