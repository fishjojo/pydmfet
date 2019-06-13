cmake_minimum_required(VERSION 2.8.12)
project(pydmfet)
include(GNUInstallDirs)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -qopenmp -xhost")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp -xhost")

add_library(hess SHARED pydmfet/libcpp/hess.cpp)
target_link_libraries(hess -lmkl_intel_thread -lmkl_core -lmkl_intel_lp64  -lmkl_avx)

add_library(svd SHARED pydmfet/libcpp/mkl_svd.cpp)
target_link_libraries(svd -lmkl_intel_thread -lmkl_core -lmkl_intel_lp64  -lmkl_avx)

install(TARGETS hess svd
	LIBRARY DESTINATION pydmfet/libcpp)

#add_subdirectory(pybind11)
#pybind11_add_module(libhess pydmfet/libcpp/hess.cpp)
