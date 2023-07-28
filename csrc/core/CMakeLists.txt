find_package(Eigen3 REQUIRED)

add_library(motion SHARED motion.cpp)
target_include_directories(motion PRIVATE ../3rd/sophus ${EIGEN3_INCLUDE_DIR})
target_link_libraries(motion PUBLIC sophus core solver_cholmod)

if(MSVC)
    add_definitions("/bigobj")
endif()
