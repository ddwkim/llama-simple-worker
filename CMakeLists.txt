cmake_minimum_required(VERSION 3.14)

set(CMAKE_VERBOSE_MAKEFILE ON)

project("llama_simple_worker" LANGUAGES CXX C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include paths for third-party libraries
include_directories(${CMAKE_SOURCE_DIR}/third-party/llama.cpp/include)
include_directories(${CMAKE_SOURCE_DIR}/third-party/llama.cpp/common)
include_directories(${CMAKE_SOURCE_DIR}/third-party/llama.cpp/ggml/include)
include_directories(${CMAKE_SOURCE_DIR}/include)


add_executable(llama_simple_worker
    llama-simple-worker.cpp
)

target_link_libraries(llama_simple_worker PRIVATE llama common)

message(STATUS "Building ggml and llama from source")

# Add the third-party llama.cpp project
add_subdirectory(${CMAKE_SOURCE_DIR}/third-party/llama.cpp)

# Link against the llama and common targets defined in llama.cpp
target_link_libraries(llama_simple_worker PRIVATE llama common)
