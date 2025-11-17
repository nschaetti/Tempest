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

#pragma once

#include "tempest/common.hpp"

/**
 * @brief GLFW callback that resizes the OpenGL viewport when the window changes.
 */
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

/**
 * @brief Keyboard handler that toggles pause/reset and closes the window.
 */
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

/**
 * @brief Compile and link a trivial shader program that draws a textured quad.
 */
GLuint create_shader_program();

/**
 * @brief Upload vertex/index data for a full screen quad and return VAO/VBO/EBO ids.
 */
void create_fullscreen_quad(GLuint& vao, GLuint& vbo, GLuint& ebo);

/**
 * @brief Convert the CUDA pressure field into RGBA colors and refresh the GL texture.
 *
 * The CUDA kernel writes into a Pixel Buffer Object (PBO) that is shared with
 * OpenGL. Once the kernel completes we update the texture via glTexSubImage2D.
 */
void update_texture_from_field(cudaGraphicsResource_t& resource,
                               float* d_field,
                               int nx,
                               int nz,
                               const dim3& color_grid,
                               const dim3& color_block,
                               float gain,
                               GLuint texture,
                               GLuint pbo);

/**
 * @brief Render the texture (containing the latest wave field) to the full window.
 */
void draw_fullscreen(GLuint program, GLuint vao, GLuint texture, int width, int height);
