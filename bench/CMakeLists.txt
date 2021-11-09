cmake_minimum_required(VERSION 3.2)
project(simdexp_bench)

add_subdirectory(lib/google/benchmark)

include_directories(${ADDITION_HEADERS_DIR})
include_directories(lib/google/benchmark/include)

set(SOURCE_FILES main.cpp src/all_benchmarks.cpp)

add_executable(all_benchmarks ${SOURCE_FILES})
target_link_libraries(all_benchmarks addition benchmark)
install(TARGETS all_benchmarks DESTINATION bin)
