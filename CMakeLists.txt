cmake_minimum_required(VERSION 3.20)
project(sycl_khr_properties LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(test_properties test/test_properties.cpp)
target_include_directories(test_properties PRIVATE include)

add_executable(test_properties_v2 test/test_properties_v2.cpp)
target_include_directories(test_properties_v2 PRIVATE include)

add_executable(test_properties_v3 test/test_properties_v3.cpp)
target_include_directories(test_properties_v3 PRIVATE include)

add_executable(test_properties_v4 test/test_properties_v4.cpp)
target_include_directories(test_properties_v4 PRIVATE include)

add_executable(test_properties_v5 test/test_properties_v5.cpp)
target_include_directories(test_properties_v5 PRIVATE include)

add_executable(test_properties_v6 test/test_properties_v6.cpp)
target_include_directories(test_properties_v6 PRIVATE include)
