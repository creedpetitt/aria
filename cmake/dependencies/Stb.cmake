function(aria_io_link_stb target)
  find_package(Stb REQUIRED)
  target_include_directories(${target} SYSTEM PRIVATE ${Stb_INCLUDE_DIR})
endfunction()
