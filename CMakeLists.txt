cmake_minimum_required(VERSION 3.21)
project(whoosh_lpr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(WHOOSH_LPR_TESTS "Build tests" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
  # OpenCV 5 moved convexHull, contourArea, getPerspectiveTransform to "geometry"
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs geometry)
endif()
find_package(CLI11 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
include(cmake/onnxruntime.cmake)

add_library(whoosh_lpr_core
  src/geometry.cpp
  src/detector.cpp
  src/reader.cpp
  src/pipeline.cpp)
target_include_directories(whoosh_lpr_core PUBLIC include)
target_link_libraries(whoosh_lpr_core PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)
target_compile_options(whoosh_lpr_core PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)

add_executable(whoosh-lpr src/main.cpp)
target_link_libraries(whoosh-lpr PRIVATE whoosh_lpr_core CLI11::CLI11 nlohmann_json::nlohmann_json spdlog::spdlog)

# Next to the binary: the runtime library and the models, so the folder can be copied as is.
add_custom_command(TARGET whoosh-lpr POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_RUNTIME_FILES} $<TARGET_FILE_DIR:whoosh-lpr>
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:whoosh-lpr>/models
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_SOURCE_DIR}/models/detector.onnx ${CMAKE_SOURCE_DIR}/models/ocr.onnx
          $<TARGET_FILE_DIR:whoosh-lpr>/models)
if(UNIX AND NOT APPLE)
  set_target_properties(whoosh-lpr PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()

if(WHOOSH_LPR_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)
  add_executable(whoosh_lpr_tests tests/test_geometry.cpp tests/test_reference.cpp)
  target_link_libraries(whoosh_lpr_tests PRIVATE whoosh_lpr_core GTest::gtest_main nlohmann_json::nlohmann_json)
  target_compile_definitions(whoosh_lpr_tests PRIVATE WHOOSH_LPR_ROOT="${CMAKE_SOURCE_DIR}")
  if(WIN32)
    add_custom_command(TARGET whoosh_lpr_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_RUNTIME_FILES} $<TARGET_FILE_DIR:whoosh_lpr_tests>)
  endif()
  include(GoogleTest)
  gtest_discover_tests(whoosh_lpr_tests DISCOVERY_TIMEOUT 60)
endif()
