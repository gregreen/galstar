cmake_minimum_required(VERSION 2.6)
PROJECT(galstar)

#
# Check requirements
#
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

### OpenMP support
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})

### libpeyton
find_package(libpeyton REQUIRED)
include_directories(${libpeyton_INCLUDE_DIR})

### GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})

### Boost
find_package(Boost REQUIRED COMPONENTS regex iostreams thread program_options)
include_directories(${Boost_INCLUDE_DIRS})


#
# Data Directories
#
set(DATADIRSUFFIX "share/${CMAKE_PROJECT_NAME}")
set(DATADIR "${CMAKE_INSTALL_PREFIX}/${DATADIRSUFFIX}")

#
# configure a header file to pass some of the CMake settings to the source
# code
#
configure_file(   
	"${PROJECT_SOURCE_DIR}/src/galstar_config.h.in"
	"${PROJECT_BINARY_DIR}/src/galstar_config.h"  
	ESCAPE_QUOTES
)
include_directories("${PROJECT_BINARY_DIR}/src")

#
# galstar executable
#
add_executable(galstar src/main.cpp src/sampler.cpp src/binner.cpp src/stats.cpp src/interpolation.cpp src/chain.cpp)

target_link_libraries(galstar dl)
target_link_libraries(galstar ${Boost_LIBRARIES})
target_link_libraries(galstar ${CFITSIO_LIBRARIES})
target_link_libraries(galstar ${GSL_LIBRARIES})
target_link_libraries(galstar ${libpeyton_LIBRARIES})

#
# install info
#
install (TARGETS galstar DESTINATION bin)
install(FILES
	data/MrLF.MSandRGB_v1.0.dat
	data/MSandRGBcolors_v1.3.dat
DESTINATION ${DATADIRSUFFIX})

