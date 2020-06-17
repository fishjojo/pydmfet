add_library(linalg SHARED svd.cpp)

set_target_properties(linalg PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
  LINK_FLAGS ${OpenMP_CXX_FLAGS})

target_link_libraries(linalg ${BLAS_LIBRARIES})

