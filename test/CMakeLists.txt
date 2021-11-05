cmake_minimum_required(VERSION 3.2)
project(simdexp_tests)

add_subdirectory(lib/googletest)

include_directories(${ADDITION_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

set(SOURCE_FILES main.cpp src/addition_tests.cpp)

add_executable(addition_tests ${SOURCE_FILES})
target_link_libraries(addition_tests addition gtest)
install(TARGETS addition_tests DESTINATION bin)
