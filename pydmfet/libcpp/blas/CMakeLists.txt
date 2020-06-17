add_library(blas SHARED svd.cpp)

set_target_properties(blas PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
  LINK_FLAGS ${OpenMP_CXX_FLAGS})

target_link_libraries(blas ${BLAS_LIBRARIES})

