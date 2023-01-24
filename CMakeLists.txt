cmake_minimum_required(VERSION 3.15)
project(Capstone)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Capstone main.cpp)
target_link_libraries(Capstone ${OpenCV_LIBS})