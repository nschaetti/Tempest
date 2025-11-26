find_package(Python3 COMPONENTS Development REQUIRED)
if(NOT Python3_EXTENSION_SUFFIX)
    set(Python3_EXTENSION_SUFFIX ".so")
endif()

add_library(pytempest MODULE bindings.cpp)
target_link_libraries(pytempest PRIVATE tempest Python3::Python)
target_include_directories(pytempest PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
set_target_properties(pytempest PROPERTIES
        PREFIX ""
        OUTPUT_NAME "pytempest"
        SUFFIX "${Python3_EXTENSION_SUFFIX}"
)
