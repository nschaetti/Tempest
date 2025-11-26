file(GLOB_RECURSE TEMPEST_SRC CONFIGURE_DEPENDS
        "core/*.cpp"
        "surfdisp/*.cpp"
        "anfwi/*.cpp"
        "cuda/*.cu"
)

add_library(tempest STATIC ${TEMPEST_SRC})

target_include_directories(tempest
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
target_compile_features(tempest PUBLIC cxx_std_20)
set_target_properties(tempest PROPERTIES CUDA_ARCHITECTURES 86)
