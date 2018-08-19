cmake_minimum_required(VERSION 2.6)
project(levenbergmarquardt)

add_compile_options(-std=c++11)

find_package(OpenCV 3.0 QUIET)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(levenbergmarquardt main.cpp)
target_link_libraries(levenbergmarquardt ${OpenCV_LIBS})

install(TARGETS levenbergmarquardt RUNTIME DESTINATION bin)
