cmake_minimum_required(VERSION 2.8.12)
project(pydmfet)

set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCXX)
  if(UNIX AND NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")

if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} VERSION_GREATER 3.0.0)
  cmake_policy(SET CMP0042 NEW)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (NOT BLAS_LIBRARIES)
find_package(BLAS)
endif()

if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
  set(HAVE_OPENMP 1)
else ()
  set(OpenMP_CXX_FLAGS " ")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h")
# to find config.h
include_directories("${PROJECT_BINARY_DIR}")

set(CMAKE_SKIP_BUILD_RPATH  True)
set(CMAKE_BUILD_WITH_INSTALL_RPATH True)

add_subdirectory(blas)
add_subdirectory(hess)
