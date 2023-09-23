cmake_minimum_required(VERSION 2.8)
project(dsol)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE relwithdebinfo)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

set(CMAKE_CXX_STANDARD 17)
# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()



find_package(OpenCV 4.4)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})


find_package(TBB REQUIRED)
find_package(fmt REQUIRED)

find_package(glog REQUIRED)
find_package(absl REQUIRED)
find_package(Boost REQUIRED) # circular buffer

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(xtensor) # reading npy

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/sv/dsol
${PROJECT_SOURCE_DIR}/sv/util 
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
sv/dsol/adjust.cpp 
sv/dsol/align.cpp 
sv/dsol/camera.cpp 
sv/dsol/direct.cpp 
sv/dsol/extra.cpp
sv/dsol/frame.cpp
sv/dsol/hessian.cpp
sv/dsol/image.cpp
sv/dsol/odom.cpp
sv/dsol/point.cpp 
sv/dsol/select.cpp 
sv/dsol/solve.cpp 
sv/dsol/stereo.cpp 
sv/dsol/viz.cpp 
sv/dsol/window.cpp 
sv/dsol/viewer.cpp 

sv/util/cmap.cpp 
sv/util/dataset.cpp
sv/util/eigen.cpp
sv/util/math.cpp
sv/util/metric.cpp
sv/util/ocv.cpp 
sv/util/summary.cpp 

sv/dsol/adjust.h 
sv/dsol/align.h 
sv/dsol/camera.h
sv/dsol/dim.h
sv/dsol/direct.h 
sv/dsol/extra.h
sv/dsol/frame.h
sv/dsol/hessian.h 
sv/dsol/image.h 
sv/dsol/odom.h 
sv/dsol/pixel.h 
sv/dsol/point.h 
sv/dsol/select.h 
sv/dsol/solve.h
sv/dsol/stereo.h 
sv/dsol/viz.h 
sv/dsol/viewer.h 
sv/dsol/window.h 
sv/util/cmap.h
sv/util/dataset.h
sv/util/eigen.h
sv/util/grid2d.h
sv/util/logging.h
sv/util/math.h 
sv/util/metric.h 
sv/util/ocv.h 
sv/util/poly.h 
sv/util/stats.h 
sv/util/summary.h 
sv/util/tbb.h 
sv/util/timer.h 

)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
-lboost_serialization
-lcrypto
glog::glog
fmt::fmt
absl::span
absl::flags_parse
absl::flat_hash_map
absl::btree
TBB::tbb
benchmark::benchmark
GTest::GTest
)

#-lglog

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

add_executable(dsol_kitti example/dsol_kitti.cpp)
target_link_libraries(dsol_kitti ${PROJECT_NAME})

