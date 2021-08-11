cmake_minimum_required(VERSION 3.4.1)

project(rknn)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(STATUS "64bit")
	set(LIB_ARCH lib64)
else()
	message(STATUS "32bit")
	set(LIB_ARCH lib)
endif()

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

# opencv
if(LIB_ARCH STREQUAL "lib")
	set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
else()
	set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
endif()
find_package(OpenCV REQUIRED)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_classfication_demo
	${CMAKE_SOURCE_DIR}/examples/rknn_classification_demo/rknn_classification.cc
)

target_link_libraries(rknn_classfication_demo
	${RKNN_API_LIB}
	${OpenCV_LIBS}
)

add_executable(rknn_identify_demo
	${CMAKE_SOURCE_DIR}/examples/rknn_identify_demo/rknn_identify.cc
)

target_link_libraries(rknn_identify_demo
	${RKNN_API_LIB}
	${OpenCV_LIBS}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/)
install(TARGETS rknn_classfication_demo DESTINATION ./)
install(TARGETS rknn_identify_demo DESTINATION ./)
install(DIRECTORY models DESTINATION ./)
install(DIRECTORY data DESTINATION ./)
install(DIRECTORY labels DESTINATION ./)
install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)
