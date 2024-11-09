cmake_minimum_required(VERSION 3.29)
project(HillisSteeleCUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(HillisSteeleCUDA main.cu)

set_target_properties(HillisSteeleCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
