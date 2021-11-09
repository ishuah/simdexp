cmake_minimum_required(VERSION 3.2)
project(simdexp)

add_subdirectory(experiments)
set(SOURCE_FILES main.cpp)

add_executable(simdexp ${SOURCE_FILES})
target_link_libraries(simdexp experiments)
install(TARGETS simdexp DESTINATION ${SIMDEXP_INSTALL_BIN_DIR})
