cmake_minimum_required(VERSION 2.8)

project (SurfaceFitting)


#include_directories(/usr/include/cminpack-1/)
SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_STANDARD 14)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()

find_package(Boost)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(tttPlaneFitting tttPlaneFitting.cpp)

target_link_libraries(tttPlaneFitting ${ITK_LIBRARIES})

add_executable(tttSphereFitting tttSphereFitting.cpp)

target_link_libraries(tttSphereFitting ${ITK_LIBRARIES})

add_executable(tttEllipsoidFitting tttEllipsoidFitting.cpp)

target_link_libraries(tttEllipsoidFitting ${ITK_LIBRARIES})

add_executable(tttDPGMMTest tttDPGMMTest.cpp)

target_link_libraries(tttDPGMMTest ${ITK_LIBRARIES})
