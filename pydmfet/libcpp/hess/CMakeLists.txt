add_library(hess SHARED hess.cpp)

set_target_properties(hess PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
  LINK_FLAGS ${OpenMP_CXX_FLAGS})

target_link_libraries(hess ${BLAS_LIBRARIES})

