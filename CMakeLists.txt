cmake_minimum_required(VERSION 3.2)

project(simdexp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -mavx2 -mfma -m64 -mf16c -ffast-math")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(SIMDEXP_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SIMDEXP_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(SIMDEXP_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(EXPERIMENTS_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/experiments)

include_directories(${SIMDEXP_INSTALL_INCLUDE_DIR})
include_directories(${EXPERIMENTS_HEADERS_DIR})

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(bench)


