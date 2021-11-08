cmake_minimum_required(VERSION 3.2)
project(simdexp_bench)

add_subdirectory(lib/google/benchmark)

include_directories(${ADDITION_HEADERS_DIR})
include_directories(lib/google/benchmark/include)

set(SOURCE_FILES main.cpp src/addition_benchmarks.cpp)

add_executable(addition_benchmarks ${SOURCE_FILES})
target_link_libraries(addition_benchmarks addition benchmark)
install(TARGETS addition_benchmarks DESTINATION bin)
