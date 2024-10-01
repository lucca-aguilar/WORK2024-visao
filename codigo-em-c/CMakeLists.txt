cmake_minimum_required(VERSION 3.0)
project(Visao)

set(CMAKE_CXX_STANDARD 17)             
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(Visao src/main.cpp)

target_link_libraries(Visao ${OpenCV_LIBS})