set(aria_io_sources
    ${PROJECT_SOURCE_DIR}/src/io/text.cpp
    ${PROJECT_SOURCE_DIR}/src/io/features.cpp)

foreach(format IN ITEMS JSON CSV IMAGE)
  if(ARIA_IO_${format})
    string(TOLOWER "${format}" source)
    list(APPEND aria_io_sources ${PROJECT_SOURCE_DIR}/src/io/${source}.cpp)
  endif()
endforeach()

add_library(aria_io STATIC ${aria_io_sources})
add_library(aria::io ALIAS aria_io)
target_link_libraries(aria_io PUBLIC aria::aria)
target_compile_features(aria_io PUBLIC cxx_std_23)
set_target_properties(aria_io PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(aria_io PRIVATE
    ARIA_IO_HAS_JSON=$<BOOL:${ARIA_IO_JSON}>
    ARIA_IO_HAS_CSV=$<BOOL:${ARIA_IO_CSV}>
    ARIA_IO_HAS_IMAGE=$<BOOL:${ARIA_IO_IMAGE}>)

if(ARIA_IO_JSON)
  include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Json.cmake)
  aria_io_link_json(aria_io)
endif()
if(ARIA_IO_CSV)
  include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Csv.cmake)
  aria_io_link_csv(aria_io)
endif()
if(ARIA_IO_IMAGE)
  include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Stb.cmake)
  aria_io_link_stb(aria_io)
endif()

if(BUILD_TESTING)
  add_executable(io_tests ${PROJECT_SOURCE_DIR}/tests/io_tests.cpp)
  target_link_libraries(io_tests PRIVATE aria::io)
  add_test(NAME io_tests COMMAND io_tests)
endif()
