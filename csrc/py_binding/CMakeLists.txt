find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++17")
pybind11_add_module(ext SHARED bind.cpp)
target_link_libraries(ext PUBLIC motion)
target_include_directories(ext PRIVATE ../ ../3rd/sophus ${EIGEN3_INCLUDE_DIR})
