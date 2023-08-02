find_package(Eigen3 REQUIRED)

add_library(motreg SHARED motreg.cpp)
target_include_directories(motreg PRIVATE ../3rd/sophus ${EIGEN3_INCLUDE_DIR})
target_link_libraries(motreg PUBLIC sophus core solver_cholmod)

if(MSVC)
    add_definitions("/bigobj")
endif()
