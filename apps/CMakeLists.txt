set(TEMPEST_APP_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/bin)

add_executable(tempest-surfdisp tempest-surfdisp.cpp)
add_executable(tempest-anfwi tempest-anfwi.cpp)
add_executable(tempest-viewer tempest-viewer.cpp)

foreach(exe tempest-surfdisp tempest-anfwi tempest-viewer)
    target_link_libraries(${exe} PRIVATE tempest)
    set_target_properties(${exe} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${TEMPEST_APP_OUTPUT_DIR})
endforeach()
