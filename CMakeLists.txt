cmake_minimum_required(VERSION 2.6)
PROJECT(galfit)

# cmake -Dlibpeyton_ROOT=/home/mjuric/projects/libpeyton/debug  ..

#
# Check requirements
#
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

### fmemopen
include (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
check_function_exists(fmemopen HAVE_FMEMOPEN)

### libpeyton
find_package(libpeyton REQUIRED)
include_directories(${libpeyton_INCLUDE_DIR})

### GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})

### Boost
find_package(Boost REQUIRED COMPONENTS regex iostreams thread)
include_directories(${Boost_INCLUDE_DIRS})



ADD_EXECUTABLE(galfit src/main.cpp)

target_link_libraries(galfit dl)

target_link_libraries(galfit ${Boost_LIBRARIES})
target_link_libraries(galfit ${CFITSIO_LIBRARIES})
target_link_libraries(galfit ${GSL_LIBRARIES})
target_link_libraries(galfit ${libpeyton_LIBRARIES})
