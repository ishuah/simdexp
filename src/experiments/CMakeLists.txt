cmake_minimum_required(VERSION 3.2)
project(experiments C CXX)

set(SOURCE_FILES
    addition.h
    addition.cpp
    multiplication.h
    multiplication.cpp
    dotproduct.h
    dotproduct.cpp
)

add_library(experiments SHARED STATIC ${SOURCE_FILES})

install(TARGETS experiments DESTINATION ${SIMDEXP_INSTALL_LIB_DIR})
install(FILES addition.h DESTINATION ${SIMDEXP_INSTALL_INCLUDE_DIR})
install(FILES multiplication.h DESTINATION ${SIMDEXP_INSTALL_INCLUDE_DIR})
install(FILES dotproduct.h DESTINATION ${SIMDEXP_INSTALL_INCLUDE_DIR})
