function(aria_io_link_csv target)
  find_package(unofficial-vincentlaucsb-csv-parser CONFIG REQUIRED)
  target_link_libraries(${target} PRIVATE unofficial::vincentlaucsb-csv-parser::csv)
endfunction()
