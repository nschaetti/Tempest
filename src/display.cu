#include "tempest/display.hpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int, int action, int) {
    auto* state = reinterpret_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (!state)
        return;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        state->paused = !state->paused;
    } else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        state->request_reset = true;
    }
}

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint log_length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<char> log(static_cast<size_t>(log_length));
        glGetShaderInfoLog(shader, log_length, nullptr, log.data());
        std::cerr << "Shader compilation failed: " << log.data() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return shader;
}

GLuint create_shader_program() {
    static const char* vertex_src = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aTexCoord;
        out vec2 vTexCoord;
        void main() {
            vTexCoord = aTexCoord;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )";

    static const char* fragment_src = R"(
        #version 330 core
        in vec2 vTexCoord;
        out vec4 FragColor;
        uniform sampler2D uTexture;
        void main() {
            FragColor = texture(uTexture, vTexCoord);
        }
    )";

    GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_src);

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        GLint log_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<char> log(static_cast<size_t>(log_length));
        glGetProgramInfoLog(program, log_length, nullptr, log.data());
        std::cerr << "Shader link failed: " << log.data() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    glUseProgram(program);
    GLint tex_loc = glGetUniformLocation(program, "uTexture");
    glUniform1i(tex_loc, 0);
    glUseProgram(0);

    return program;
}

void create_fullscreen_quad(GLuint& vao, GLuint& vbo, GLuint& ebo) {
    const float vertices[] = {
        // positions   // texcoords
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
    };

    const unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
}

__global__ void wave_to_color_kernel(float4* rgba, const float* p, int nx, int nz, float gain) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iz >= nz)
        return;

    int idx = iz * nx + ix;
    float value = p[idx] * gain;
    value = fmaxf(fminf(value, 1.0f), -1.0f);

    float4 color;
    if (value >= 0.0f) {
        color = make_float4(value, 0.0f, 1.0f - value, 1.0f);
    } else {
        float mag = -value;
        color = make_float4(0.0f, 0.0f, mag, 1.0f);
    }
    rgba[idx] = color;
}

void update_texture_from_field(cudaGraphicsResource_t& resource,
                               float* d_field,
                               int nx,
                               int nz,
                               const dim3& color_grid,
                               const dim3& color_block,
                               float gain,
                               GLuint texture,
                               GLuint pbo) {
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource, 0));
    float4* d_output = nullptr;
    size_t num_bytes = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_output),
                                                    &num_bytes, resource));

    wave_to_color_kernel<<<color_grid, color_block>>>(d_output, d_field, nx, nz, gain);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, 0));

    glBindTexture(GL_TEXTURE_2D, texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, nz, GL_RGBA, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void draw_fullscreen(GLuint program, GLuint vao, GLuint texture, int width, int height) {
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
    glUseProgram(0);
}
