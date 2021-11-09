cmake_minimum_required(VERSION 3.2)
project(simdexp_tests)

add_subdirectory(lib/googletest)

include_directories(${EXPERIMENTS_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

set(SOURCE_FILES main.cpp src/all_tests.cpp)

add_executable(all_tests ${SOURCE_FILES})
target_link_libraries(all_tests experiments gtest)
install(TARGETS all_tests DESTINATION bin)
