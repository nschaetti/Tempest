#include "tempest/display.hpp"
#include "tempest/init.hpp"
#include "tempest/simulation.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return EXIT_FAILURE;
    }

    SimulationConfig cfg = load_config(argv[1]);

    if (cfg.block_size_x <= 0 || cfg.block_size_y <= 0)
        throw std::runtime_error("Block sizes must be positive");
    if (cfg.block_size_x > BLOCK_SIZE_X || cfg.block_size_y > BLOCK_SIZE_Y)
        throw std::runtime_error("Configured block size exceeds compiled BLOCK_SIZE limits");
    if (cfg.display_interval <= 0)
        throw std::runtime_error("display_interval must be >= 1");
    if (cfg.display_scale <= 0.0f)
        throw std::runtime_error("display_scale must be > 0");

    print_config(cfg);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    int window_width = std::max(static_cast<int>(cfg.nx * cfg.display_scale), 100);
    int window_height = std::max(static_cast<int>(cfg.nz * cfg.display_scale), 100);

    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Tempest CUDA", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    glewExperimental = GL_TRUE;
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(glew_status) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glDisable(GL_DEPTH_TEST);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    InputState input_state{};
    glfwSetWindowUserPointer(window, &input_state);
    glfwSetKeyCallback(window, key_callback);

    GLuint shader_program = create_shader_program();
    GLuint quad_vao = 0, quad_vbo = 0, quad_ebo = 0;
    create_fullscreen_quad(quad_vao, quad_vbo, quad_ebo);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, cfg.nx, cfg.nz, 0, GL_RGBA, GL_FLOAT, nullptr);

    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<size_t>(cfg.nx) * cfg.nz * sizeof(float4), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsResource_t cuda_pbo_resource = nullptr;
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices available");
    }
    int cuda_device = 0;
    CUDA_CHECK(cudaGLSetGLDevice(cuda_device));
    CUDA_CHECK(cudaSetDevice(cuda_device));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

    const int nx = cfg.nx;
    const int nz = cfg.nz;
    const int n = nx * nz;
    const float dt2dx2 = (cfg.dt * cfg.dt) / (cfg.dx * cfg.dx);

    float *d_p_old = nullptr, *d_p = nullptr, *d_p_new = nullptr, *d_c2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p_old, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_new, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2, n * sizeof(float)));

    int source_ix = nx / 2;
    int source_iz = nz / 2;
    int source_index = source_iz * nx + source_ix;
    initialize_wavefields(d_p_old, d_p, d_p_new, d_c2, cfg.c0 * cfg.c0, n, source_index, 1.0f);

    dim3 block(cfg.block_size_x, cfg.block_size_y);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);

    dim3 color_block(16, 16);
    dim3 color_grid((nx + color_block.x - 1) / color_block.x,
                    (nz + color_block.y - 1) / color_block.y);

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt, 0));

    int steps_completed = 0;
    bool texture_dirty = true;

    auto upload_frame = [&]() {
        update_texture_from_field(cuda_pbo_resource, d_p, nx, nz,
                                  color_grid, color_block, 5.0f,
                                  texture, pbo);
        texture_dirty = false;
    };

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (input_state.request_reset) {
            reset_wavefields(d_p_old, d_p, d_p_new, n, source_index, 1.0f);
            input_state.request_reset = false;
            steps_completed = 0;
            texture_dirty = true;
            CUDA_CHECK(cudaEventRecord(start_evt, 0));
        }

        bool simulation_active = !input_state.paused && steps_completed < cfg.nt;

        if (simulation_active) {
            wave_step_kernel<<<grid, block>>>(d_p_new, d_p, d_p_old, d_c2, nx, nz, dt2dx2);
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_p_old, d_p);
            std::swap(d_p, d_p_new);
            ++steps_completed;
            if (steps_completed % cfg.display_interval == 0 || steps_completed == cfg.nt) {
                texture_dirty = true;
            }
        }

        if (texture_dirty) {
            upload_frame();
        }

        int fb_width = 0;
        int fb_height = 0;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        fb_width = std::max(fb_width, 1);
        fb_height = std::max(fb_height, 1);

        draw_fullscreen(shader_program, quad_vao, texture, fb_width, fb_height);
        glfwSwapBuffers(window);

        if (steps_completed >= cfg.nt) {
            break;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_evt, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_evt, stop_evt));
    if (steps_completed == 0) {
        steps_completed = 1;
    }
    std::cout << "Simulation finished: " << steps_completed << " steps" << std::endl;
    std::cout << "Total GPU time: " << milliseconds << " ms" << std::endl;
    std::cout << "Average per step: " << (milliseconds / steps_completed) << " ms" << std::endl;

    if (cuda_pbo_resource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    }
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    glDeleteProgram(shader_program);
    glDeleteBuffers(1, &quad_vbo);
    glDeleteBuffers(1, &quad_ebo);
    glDeleteVertexArrays(1, &quad_vao);

    glfwDestroyWindow(window);
    glfwTerminate();

    CUDA_CHECK(cudaFree(d_p_old));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_new));
    CUDA_CHECK(cudaFree(d_c2));

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));

    return 0;
}
