
/*
 * This file is part of Tempest.
 *
 * Tempest is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Tempest is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Tempest.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "tempest/display.hpp"
#include "tempest/init.hpp"
#include "tempest/simulation.hpp"

// Entry point that wires together configuration loading, GPU allocations,
// OpenGL rendering, and the main simulation loop.

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return EXIT_FAILURE;
    }

    // Load all numeric parameters so CUDA/OpenGL know how big to allocate things.
    SimulationConfig cfg = load_config(argv[1]);

    // Sanity-check user-provided settings before touching any GPU state.
    if (cfg.block_size_x <= 0 || cfg.block_size_y <= 0)
        throw std::runtime_error("Block sizes must be positive");
    if (cfg.block_size_x > BLOCK_SIZE_X || cfg.block_size_y > BLOCK_SIZE_Y)
        throw std::runtime_error("Configured block size exceeds compiled BLOCK_SIZE limits");
    if (cfg.display_interval <= 0)
        throw std::runtime_error("display_interval must be >= 1");
    if (cfg.display_scale <= 0.0f)
        throw std::runtime_error("display_scale must be > 0");

    print_config(cfg);

    // Initialise GLFW (gestion des fenêtres + contexte OpenGL + input).
    // glfwInit() retourne GLFW_TRUE si succès, 0 si échec.
    if (!glfwInit()) {
        // Si glfwInit() a retourné 0, on affiche un message d’erreur sur la sortie d’erreur.
        std::cerr << "Failed to initialize GLFW" << std::endl;

        // On quitte immédiatement le programme avec le code standard d’échec.
        return EXIT_FAILURE;
    }

    // ---------------------------------------------------------------
    // Demande à GLFW de créer un contexte OpenGL moderne (version 3.3).
    // Ces "hints" doivent être placés *avant* glfwCreateWindow.
    // ---------------------------------------------------------------

    // Version majeure du contexte OpenGL demandé : 3.x
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);

    // Version mineure du contexte OpenGL demandé : x.3 → donc 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    // Demande le "Core Profile" : supprime toutes les vieilles fonctions OpenGL
    // (glBegin(), glVertex*, etc.) et force l'utilisation du pipeline moderne.
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Window dimensions are derived from the simulation grid so one texel ~ one cell.
    // On multiplie le nombre de cellules en X par le facteur d'échelle pour obtenir
    // une largeur de fenêtre en pixels. Comme la multiplication produit un float,
    // on utilise static_cast<int> pour convertir explicitement en entier.
    //
    // ⚠️ Pourquoi static_cast<int> ?
    // - C’est la méthode C++ MODERNE pour convertir un type vers un autre.
    // - Ici cfg.nx (int) * cfg.display_scale (float) donne un float.
    // - GLFW exige des dimensions en pixels → donc un int.
    // - static_cast<int>(...) convertit proprement le float en int.
    // - Contrairement à (int)(...) du C, static_cast<> est plus sûr, plus clair,
    //   et évite des conversions dangereuses.
    //
    // std::max(..., 100) garantit une largeur minimum de 100 pixels.
    int window_width = std::max(static_cast<int>(cfg.nx * cfg.display_scale), 100);

    // Même logique pour la hauteur :
    // - cfg.nz * cfg.display_scale → float
    // - static_cast<int> → conversion contrôlée en entier
    // - std::max(..., 100) → minimum 100 px
    int window_height = std::max(static_cast<int>(cfg.nz * cfg.display_scale), 100);

    // Création de la fenêtre GLFW avec les dimensions calculées.
    // Arguments :
    //   - window_width  : largeur de la fenêtre en pixels
    //   - window_height : hauteur de la fenêtre en pixels
    //   - "Tempest CUDA": titre affiché dans la barre de la fenêtre
    //   - nullptr       : pas de monitor → donc mode fenêtré classique (pas fullscreen)
    //
    //   Dernier argument : le *contexte OpenGL à partager*  (share)
    //   -----------------------------------------------------
    //   Le paramètre final de glfwCreateWindow est un pointeur vers une autre
    //   fenêtre GLFW dont on souhaite *partager le contexte OpenGL*.
    //
    //   🔹 C’est quoi un contexte OpenGL ?
    //   - C’est un environnement complet contenant TOUT l’état OpenGL :
    //       * shaders compilés
    //       * textures
    //       * VBO/VAO
    //       * programmes GPU
    //       * framebuffers
    //       * objets OpenGL en général
    //   - Chaque fenêtre GLFW possède son propre contexte.
    //   - Un contexte = la “session” OpenGL d'une fenêtre.
    //
    //   🔹 C’est quoi *partager un contexte* ?
    //   - Si on passe une autre fenêtre comme argument `share`, alors :
    //       * les deux fenêtres utilisent le même contexte OpenGL,
    //       * elles partagent donc les mêmes textures,
    //       * les mêmes shaders,
    //       * les mêmes buffers,
    //       * les mêmes objets GPU,
    //       * les mêmes ressources.
    //   - Cela permet par exemple :
    //       * d’avoir deux fenêtres affichant la *même* scène 3D,
    //       * d’afficher une UI dans une fenêtre et un rendu dans l’autre,
    //       * de faire du rendu off-screen dans un contexte et de l’afficher ailleurs.
    //
    //   🔹 Pourquoi ici on met nullptr ?
    //   - Parce que Tempest utilise une seule fenêtre.
    //   - Aucun besoin de partager un contexte OpenGL avec une autre fenêtre.
    //   - Donc : *ce contexte n'est partagé avec personne*.
    //
    //   En résumé :
    //       nullptr = "ne partage pas ce contexte OpenGL avec d’autres fenêtres".
    //
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Tempest CUDA", nullptr, nullptr);

    // Vérifie que la création de la fenêtre a réussi (sinon glfwCreateWindow retourne nullptr).
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;

        // Libère les ressources GLFW avant de quitter.
        glfwTerminate();
        return EXIT_FAILURE;
    }

    // Associe le contexte OpenGL de cette fenêtre au thread courant.
    // Toute commande OpenGL après cette ligne affectera *cette* fenêtre.
    glfwMakeContextCurrent(window);

    // Désactive la synchronisation verticale (vsync).
    // Sans vsync : la simulation tourne à pleine vitesse, non limitée par 60 Hz.
    glfwSwapInterval(0);

    // Active l’accès aux extensions modernes OpenGL pour GLEW.
    // Certains drivers nécessitent ce flag pour exposer les fonctions récentes.
    glewExperimental = GL_TRUE;

        // Initialise GLEW pour charger dynamiquement l’ensemble des fonctions OpenGL.
    // ⚠️ Cette fonction doit être appelée APRÈS glfwMakeContextCurrent.
    GLenum glew_status = glewInit();

    // Vérifie que GLEW a bien chargé toutes les extensions OpenGL requises.
    if (glew_status != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: "
                  << glewGetErrorString(glew_status) << std::endl;

        // Détruit la fenêtre créée et ferme GLFW proprement.
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    // Associe le contexte OpenGL de la fenêtre courante au thread actuel.
    // Toute commande OpenGL après cette ligne agira sur *cette* fenêtre.
    glfwMakeContextCurrent(window);

    // Désactive la synchronisation verticale (vsync).
    // Cela empêche le GPU d’attendre la fréquence de rafraîchissement du moniteur.
    // Résultat : la simulation est rendue aussi vite que possible.
    glfwSwapInterval(0);

    // Indique à GLEW d'activer l'accès aux extensions modernes d’OpenGL.
    // Nécessaire sur certains drivers (surtout Linux + NVIDIA).
    glewExperimental = GL_TRUE;

    // Initialise GLEW pour charger dynamiquement toutes les fonctions OpenGL.
    // glewInit() doit être appelé *après* glfwMakeContextCurr

    glDisable(GL_DEPTH_TEST); // We draw a flat quad, so depth buffering is unnecessary.
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    InputState input_state{};
    // Store the pointer inside GLFW so callbacks can flip flags when keys are pressed.
    glfwSetWindowUserPointer(window, &input_state);
    glfwSetKeyCallback(window, key_callback);

    // One simple shader + quad is enough to display the pressure field texture.
    GLuint shader_program = create_shader_program();
    GLuint quad_vao = 0, quad_vbo = 0, quad_ebo = 0;
    create_fullscreen_quad(quad_vao, quad_vbo, quad_ebo);

    // Floating-point texture stores RGBA colors generated by CUDA.
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, cfg.nx, cfg.nz, 0, GL_RGBA, GL_FLOAT, nullptr);

    // Pixel Buffer Object (PBO) is the shared memory bridge between CUDA and OpenGL.
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
    // Register the OpenGL buffer so CUDA kernels can write directly into it.
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

    const int nx = cfg.nx;
    const int nz = cfg.nz;
    const int n = nx * nz;
    // Pre-compute dt^2/dx^2 once to keep the kernel inner loop tiny.
    const float dt2dx2 = (cfg.dt * cfg.dt) / (cfg.dx * cfg.dx);

    // Allocate three wavefield buffers (previous, current, next) plus c^2.
    float *d_p_old = nullptr, *d_p = nullptr, *d_p_new = nullptr, *d_c2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p_old, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_new, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2, n * sizeof(float)));

    int source_ix = nx / 2;
    int source_iz = nz / 2;
    int source_index = source_iz * nx + source_ix;
    // Seed the wavefield with a single impulse in the center of the domain.
    initialize_wavefields(d_p_old, d_p, d_p_new, d_c2, cfg.c0 * cfg.c0, n, source_index, 1.0f);

    dim3 block(cfg.block_size_x, cfg.block_size_y);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);

    // Separate launch configuration for the color-conversion kernel.
    dim3 color_block(16, 16);
    dim3 color_grid((nx + color_block.x - 1) / color_block.x,
                    (nz + color_block.y - 1) / color_block.y);

    // Events measure total GPU simulation time.
    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt, 0));

    int steps_completed = 0;
    bool texture_dirty = true; // Force an initial upload before drawing anything.

    // Small helper so we can mark the texture dirty and refresh on demand.
    auto upload_frame = [&]() {
        update_texture_from_field(cuda_pbo_resource, d_p, nx, nz,
                                  color_grid, color_block, 5.0f,
                                  texture, pbo);
        texture_dirty = false;
    };

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents(); // Process keyboard/window events once per frame.

        if (input_state.request_reset) {
            // Resetting clears the wavefields and restarts the timer.
            reset_wavefields(d_p_old, d_p, d_p_new, n, source_index, 1.0f);
            input_state.request_reset = false;
            steps_completed = 0;
            texture_dirty = true;
            CUDA_CHECK(cudaEventRecord(start_evt, 0));
        }

        bool simulation_active = !input_state.paused && steps_completed < cfg.nt;

        if (simulation_active) {
            // Launch one wave propagation step and rotate the ping-pong buffers.
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
        fb_height = std::max(fb_height, 1); // Guard against minimized windows.

        draw_fullscreen(shader_program, quad_vao, texture, fb_width, fb_height);
        glfwSwapBuffers(window);

        if (steps_completed >= cfg.nt) {
            break;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all kernels completed before timing stops.
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
