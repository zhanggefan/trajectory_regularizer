add_subdirectory(3rd)

add_subdirectory(core)

find_package(Python3 COMPONENTS Interpreter Development)
if (Python3_EXECUTABLE)
    message("-- use python at " ${Python3_EXECUTABLE})
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())" OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (pybind11_DIR)
        message("-- use pybind11 at " ${pybind11_DIR})
        find_package(pybind11 REQUIRED)
        add_subdirectory(py_binding)
    endif ()
endif ()
