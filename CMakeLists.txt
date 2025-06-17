cmake_minimum_required(VERSION 3.10)
project(Checkmate)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable CMake integration with vcpkg
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

# Find packages
find_package(OpenCV REQUIRED)

# Set the source file
set(SOURCE_FILES 
    src/calibrator.cpp
    src/chessboard.cpp
    src/frame_loader.cpp
    src/main.cpp
    src/renderer.cpp
    src/utils.cpp
)

# Add the executable
add_executable(Checkmate ${SOURCE_FILES})

# Link OpenCV libraries
target_link_libraries(Checkmate PRIVATE ${OpenCV_LIBS})

# Set output directory for the executable
set_target_properties(Checkmate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../build")

# Copy frames to the output target directory after build
add_custom_command(TARGET Checkmate POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/res/frames" "$<TARGET_FILE_DIR:Checkmate>/res/frames"
)
