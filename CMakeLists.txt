cmake_minimum_required(VERSION 3.11)
project(py_neural_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -Wno-deprecated-declarations")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(pybind11_DIR /home/hdrlab/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)

set(MKL_INTERFACE_FULL intel_lp64)

add_definitions(-DEIGEN_USE_MKL_ALL)

# Per https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html eigen should be used
# with MKL LAPACK only with lp64 set. The default may be ILP64, which we don't want.
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
find_package(MKL REQUIRED)
find_package(pybind11 REQUIRED)
find_package(SymEngine REQUIRED)

set(SOURCES
        src/main.cpp
        src/world.cpp
        src/dampingForce.cpp
        src/elasticBendingForce.cpp
        src/elasticStretchingForce.cpp
        src/elasticTwistingForce.cpp
        src/externalGravityForce.cpp
        src/inertialForce.cpp
        src/setInput.cpp
        src/timeStepper.cpp
        src/world.cpp
        src/elasticRod.cpp
        src/collisionDetector.cpp
        src/contactPotentialIMC.cpp
        src/solver.cpp
        src/bendingCompute.cpp
)

set(LINK_LIBRARIES
        ${SYMENGINE_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        MKL::MKL
        Eigen3::Eigen
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        lapack
        stdc++fs
        Threads::Threads
        m
        pybind11::module
)

if (OpenMP_CXX_FOUND)
    set (LINK_LIBRARIES ${LINK_LIBRARIES}
            OpenMP::OpenMP_CXX
    )
endif ()

add_library(common_sources STATIC ${SOURCES})

target_include_directories(common_sources PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_include_directories(common_sources SYSTEM PRIVATE
        Eigen3::Eigen
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS}
        ${SYMENGINE_INCLUDE_DIRS}
        )

target_link_libraries(common_sources PUBLIC
        ${LINK_LIBRARIES}
)

pybind11_add_module(nn_der
        src/app.cpp
)
target_link_libraries(nn_der PRIVATE
        common_sources
)

set_target_properties(nn_der PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/nn_der"
)
