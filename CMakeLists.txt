cmake_minimum_required(VERSION 3.10)
project(mini_dl)

if(MSVC)
    add_compile_options("/utf-8") # 强制源码和输出使用 UTF-8
endif()

add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# ===== Library =====
file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(mini_dl STATIC ${SRC_FILES})

set_target_properties(mini_dl PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/lib"
)

# ===== Tests =====
file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")

foreach(TEST_SRC ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} mini_dl)

    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin"
    )
endforeach()
