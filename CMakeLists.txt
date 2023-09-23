cmake_minimum_required(VERSION 3.25)
project(RWKV-NCNN)

set(BIN_NAME "chat_rwkv_ncnn")

set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn
        SYSTEM
        )
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_PIXEL "" OFF)
    option(NCNN_PIXEL_ROTATE "" OFF)
    option(NCNN_PIXEL_AFFINE "" OFF)
    option(NCNN_PIXEL_DRAWING "" OFF)
    option(NCNN_DISABLE_EXCEPTION "" OFF)
    option(NCNN_OPENMP "" ON)
    FetchContent_MakeAvailable(ncnn)
else()
    FIND_PACKAGE(ncnn REQUIRED)
endif()

FetchContent_Declare(
        msgpack
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c
        GIT_TAG cpp-6.1.0
        SYSTEM
        )
option(MSGPACK_USE_BOOST "" OFF)
FetchContent_MakeAvailable(msgpack)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

FILE(GLOB_RECURSE sourcefiles "src/*.cpp")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O${OPTIMIZATION_LEVEL}")

IF(DEBUG)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    ADD_COMPILE_DEFINITIONS(DEBUG)
ENDIF()

ADD_EXECUTABLE(${BIN_NAME} ${sourcefiles})

TARGET_LINK_LIBRARIES(${BIN_NAME} ncnn msgpack-cxx)
