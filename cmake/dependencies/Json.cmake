function(aria_io_link_json target)
  find_package(nlohmann_json 3.11 CONFIG REQUIRED)
  target_link_libraries(${target} PRIVATE nlohmann_json::nlohmann_json)
endfunction()
