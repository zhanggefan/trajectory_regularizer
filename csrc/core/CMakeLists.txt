find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++17")
add_library(motion SHARED motion.cpp)
target_include_directories(motion PRIVATE ../3rd/sophus ${EIGEN3_INCLUDE_DIR})
target_link_libraries(motion PUBLIC sophus core)
