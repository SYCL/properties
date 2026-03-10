cmake_minimum_required(VERSION 3.20)
project(sycl_khr_properties LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(test_properties test/test_properties.cpp)
target_include_directories(test_properties PRIVATE include)
