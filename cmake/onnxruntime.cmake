# ONNX Runtime: an installed package if there is one (conda-forge ships a CMake
# config), else the official prebuilt release (building it from source takes an
# hour). Set ORT_ROOT to use an unpacked release instead of downloading it.
set(ORT_VERSION 1.22.0)

# The files to put next to the binary. On Linux that is the library under its
# soname (libonnxruntime.so.1): that name, not libonnxruntime.so, is what the
# binary asks the loader for.
function(ort_runtime_files libdir)
  if(WIN32)
    file(GLOB _files ${libdir}/onnxruntime*.dll ${libdir}/../bin/onnxruntime*.dll)
  elseif(APPLE)
    file(GLOB _files ${libdir}/libonnxruntime.*.dylib)
  else()
    file(GLOB _files ${libdir}/libonnxruntime.so.[0-9]*)
    list(SORT _files COMPARE NATURAL)
    list(GET _files 0 _files)  # libonnxruntime.so.1 sorts before libonnxruntime.so.1.22.0
  endif()
  set(ORT_RUNTIME_FILES ${_files} PARENT_SCOPE)
endfunction()
if(NOT ORT_ROOT)
  find_package(onnxruntime CONFIG QUIET)
endif()
if(onnxruntime_FOUND)
  # its headers live in include/onnxruntime/core/session, the release has them flat
  get_target_property(_ort_inc onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
  list(GET _ort_inc 0 _ort_inc)
  if(EXISTS ${_ort_inc}/core/session/onnxruntime_cxx_api.h)
    set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${_ort_inc}/core/session)
  endif()
  message(STATUS "ONNX Runtime ${onnxruntime_VERSION}: installed package")
  get_target_property(_ort_loc onnxruntime::onnxruntime IMPORTED_LOCATION_RELEASE)
  if(NOT _ort_loc)
    get_target_property(_ort_loc onnxruntime::onnxruntime IMPORTED_LOCATION)
  endif()
  get_filename_component(_ort_libdir ${_ort_loc} DIRECTORY)
  ort_runtime_files(${_ort_libdir})
  return()
endif()
if(NOT ORT_ROOT)
  if(WIN32)
    set(_ort_pkg onnxruntime-win-x64-${ORT_VERSION}.zip)
  elseif(APPLE)
    set(_ort_pkg onnxruntime-osx-universal2-${ORT_VERSION}.tgz)
  else()
    set(_ort_pkg onnxruntime-linux-x64-${ORT_VERSION}.tgz)
  endif()
  include(FetchContent)
  FetchContent_Declare(onnxruntime_bin
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ort_pkg}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnxruntime_bin)
  set(ORT_ROOT ${onnxruntime_bin_SOURCE_DIR})
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
target_include_directories(onnxruntime::onnxruntime INTERFACE ${ORT_ROOT}/include)
if(WIN32)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_IMPLIB ${ORT_ROOT}/lib/onnxruntime.lib
    IMPORTED_LOCATION ${ORT_ROOT}/lib/onnxruntime.dll)
elseif(APPLE)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES IMPORTED_LOCATION ${ORT_ROOT}/lib/libonnxruntime.dylib)
else()
  set_target_properties(onnxruntime::onnxruntime PROPERTIES IMPORTED_LOCATION ${ORT_ROOT}/lib/libonnxruntime.so)
endif()
ort_runtime_files(${ORT_ROOT}/lib)
