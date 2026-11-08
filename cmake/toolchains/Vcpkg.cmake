find_program(ARIA_VCPKG_EXECUTABLE NAMES vcpkg REQUIRED)
file(REAL_PATH "${ARIA_VCPKG_EXECUTABLE}" aria_vcpkg_executable)
get_filename_component(aria_vcpkg_root "${aria_vcpkg_executable}" DIRECTORY)
set(aria_vcpkg_toolchain "${aria_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")

if(NOT EXISTS "${aria_vcpkg_toolchain}")
  message(FATAL_ERROR "The vcpkg executable was found, but its toolchain is missing: ${aria_vcpkg_toolchain}")
endif()

include("${aria_vcpkg_toolchain}")
