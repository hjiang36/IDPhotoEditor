import ctypes
from typing import Optional, Tuple
import glfw
import OpenGL.GL as gl

import imgui
from imgui.integrations.glfw import GlfwRenderer
import plyer
import numpy as np
from importlib import resources

from .data_structs import ImageDataStructs

"""
Create window with GLFW.

@param width: width of window in pixels.
@param height: height of window in pixels.
@param name: window name string.

@returns: window handle
"""
def _create_glfw_window(width: int, height: int, name: str=""):
    assert glfw.init(), "Failed to initialize GLFW context"
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    # glfw.window_hint(glfw.DECORATED, glfw.FALSE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(width, height, name, None, None)
    glfw.make_context_current(window)
    if not window:
        glfw.terminate()
        print("GLFW Could not initialize Window")
        exit(1)
    return window


"""
Wrapper for OpenGL Texture.
"""
class _Texture:
    """
    Initialize the texture class.

    @param data: data to be loaded into texture (HWC format).
    """
    def __init__(self, data: np.ndarray) -> None:
        # Check input format and data type.
        assert data is not None, "Invalid data for texture"
        assert data.dtype == np.float32 or data.dtype == np.uint8, "Invalid data type for texture"
        assert data.ndim == 2 or data.ndim == 3, "Invalid data format for texture"
        data_format = gl.GL_RED
        if data.ndim == 3:
            assert data.shape[2] == 3 or data.shape[2] == 4, "Invalid data format for texture"
            data_format = gl.GL_RGBA if data.shape[2] == 4 else gl.GL_RGB
        data_type = gl.GL_FLOAT if data.dtype == np.float32 else gl.GL_UNSIGNED_BYTE
        
        height, width = data.shape[:2]
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            data_format,
            data_type,
            data,
        )
        self._handle = texture
        self._data = data

    """
    Get the texture handle.
    """
    def id(self) -> int:
        return self._handle

    """
    Get height and width of texture.
    """
    def size(self) -> Tuple[int, int]:
        return self._data.shape[:2]

    """
    Get width of texture.
    """
    def width(self) -> int:
        return self._data.shape[1]

    """
    Get height of texture.
    """
    def height(self) -> int:
        return self._data.shape[0]

    """
    Get the texture sub image data.
    """
    def update_sub_image(self, data: np.ndarray, x: int, y: int) -> None:
        assert data is not None, "Invalid data for texture"
        assert data.dtype == np.float32 or data.dtype == np.uint8, "Invalid data type for texture"
        assert data.ndim == 2 or data.ndim == 3, "Invalid data format for texture"
        data_format = gl.GL_RED
        if data.ndim == 3:
            assert data.shape[2] == 3 or data.shape[2] == 4, "Invalid data format for texture"
            data_format = gl.GL_RGBA if data.shape[2] == 4 else gl.GL_RGB
        data_type = gl.GL_FLOAT if data.dtype == np.float32 else gl.GL_UNSIGNED_BYTE

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._handle)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            x,
            y,
            data.shape[1],
            data.shape[0],
            data_format,
            data_type,
            data,
        )

    """
    Delete the texture.
    """
    def __delete__(self) -> None:
        gl.glDeleteTextures(1, [self._handle])

"""
Wrapper for OpenGL Shader.
"""
class _Shader:
    """
    Initialize the shader class.

    @param source: shader source code.
    @param shader_type: shader type (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER).
    """
    def __init__(self, source: str, shader_type: int) -> None:
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)

        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(shader))

        self._handle = shader

    """
    Get the shader handle.
    """
    def id(self) -> int:
        return self._handle

    """
    Delete the shader.
    """
    def __delete__(self) -> None:
        gl.glDeleteShader(self._handle)

"""
Wrapper for OpenGL Shader Program.
"""
class _Program:
    """
    Initialize the program class.

    @param vertex_shader: vertex shader source code.
    @param fragment_shader: fragment shader source code.
    """
    def __init__(self, vertex_shader: str, fragment_shader: str) -> None:
        self._handle = gl.glCreateProgram()
        self._vertex_shader = _Shader(vertex_shader, gl.GL_VERTEX_SHADER)
        self._fragment_shader = _Shader(fragment_shader, gl.GL_FRAGMENT_SHADER)

        gl.glAttachShader(self._handle, self._vertex_shader.id())
        gl.glAttachShader(self._handle, self._fragment_shader.id())
        gl.glLinkProgram(self._handle)

        if not gl.glGetProgramiv(self._handle, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(self._handle))

    """
    Get the program handle.
    """
    def id(self) -> int:
        return self._handle

    """
    Delete the program.
    """
    def __delete__(self) -> None:
        self._vertex_shader.delete()
        self._fragment_shader.delete()
        gl.glDeleteProgram(self._handle)


"""
Wrapper for OpenGL Vertex Array Object.
"""
class _VertexArray:
    """
    Initialize the vertex array class.

    @param vertices: vertices to be loaded into vertex array.
    """
    def __init__(self, vertices: np.ndarray) -> None:
        self._vao = gl.glGenVertexArrays(1)
        self._vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices, gl.GL_STATIC_DRAW)

        # Position and texture coordinates attributes.
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)

    """
    Get the vertex array handle.
    """
    def id(self) -> int:
        return self._vao

    """
    Delete the vertex array.
    """
    def __delete__(self) -> None:
        gl.glDeleteVertexArrays(1, [self._vao])
        gl.glDeleteBuffers(1, [self._vbo])


"""
Image renderer class.

This class is repsponsible for rendering the image and mask.
"""
class _ImageRenderer:
    """
    Initialize the ImageRenderer class.
    We would initialize the graphics resources (texture, shader etc.) here.

    @param image: image to be rendered in HWC format.
    """
    def __init__(self, image: np.ndarray) -> None:
        self.texture = _Texture(image)
        self.mask_texture = _Texture(np.ones_like(image))
        self.mask_color = (0.0, 0.0, 0.0, 0.5)
        vertex_shader_str = resources.files("util.shaders").joinpath(
            "full_screen.vs").open("r").read()
        fragment_shader_str = resources.files("util.shaders").joinpath(
            "full_screen.fs").open("r").read()
        self.program = _Program(
            vertex_shader=vertex_shader_str,
            fragment_shader=fragment_shader_str
        )
        self.vao = _VertexArray(np.array([
            1.0,  3.0, 0.0,   1.0, -1.0, # top right
            1.0, -1.0, 0.0,   1.0, 1.0, # bottom right
            -3.0, -1.0, 0.0,   -1.0, 1.0, # bottom left
        ]).astype(np.float32))

    """
    Update the mask data.

    @param mask: mask to be rendered in HWC format.
    """
    def update_mask(self, mask: np.ndarray) -> None:
        # TODO: we might consider re-use the mask texture here.
        self.mask_texture = _Texture(mask)

    """
    Update mask sub-image data.
    """
    def update_mask_sub_image(self, new_data: np.ndarray, x: int, y: int) -> None:
        self.mask_texture.update_sub_image(new_data, x, y)

    """
    Set the mask color.

    @param color: mask color in RGBA format.
    """
    def set_mask_color(self, color: Tuple[float, float, float]) -> None:
        assert len(color) == 4, "Mask color should be in RGB format."
        self.mask_color = color

    """
    Render the image to the screen.

    @param brush_info: brush information in (x, y, radius) format.
    """
    def render(self, brush_info: Tuple[float, float, float]) -> None:
        gl.glUseProgram(self.program.id())
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.id())
        gl.glUniform1i(gl.glGetUniformLocation(self.program.id(), "tex"), 0)
        gl.glUniform2f(gl.glGetUniformLocation(self.program.id(), "texSize"),
            self.texture.width(), self.texture.height())

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mask_texture.id())
        gl.glUniform1i(gl.glGetUniformLocation(self.program.id(), "maskTexture"), 1)
        gl.glUniform4f(gl.glGetUniformLocation(
            self.program.id(), "maskColor"), *self.mask_color)

        # Brush information.
        gl.glUniform3f(gl.glGetUniformLocation(self.program.id(), "brushInfo"), *brush_info)

        gl.glBindVertexArray(self.vao.id())
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

"""
The GUI renderer that the user interact with.
"""
class GUIRenderer:
    """
    Initialize the GUIRenderer class.
    This call would initialize the ImGUI, graphics context and create
    window to be ready for actual rendering.

    @param width: width of window in pixels.
    @param height: height of window in pixels.
    @param name: name of the window
    """
    def __init__(self, width: int, height: int, name: str) -> None:
        imgui.create_context()
        self.window = _create_glfw_window(width, height, name)
        self.impl = GlfwRenderer(self.window)
        self.image_renderer: Optional[_ImageRenderer] = None
        self.window_width = width
        self.window_height = height
        self.mask_color = (0.0, 0.0, 0.0, 0.5)

        self.edit_mode_index = 0
        self.edit_mode_options = ["Move", "Brush"]
        self.brush_radius = 20.0

    
    """
    Whether window should be closed.

    @return: bool, if window should be closed.
    """
    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)

    """
    Render one frame.
    TODO: we might want to register a controller here.

    @param 
    """
    def render(self, data: ImageDataStructs) -> None:
        glfw.poll_events()
        self.impl.process_inputs()

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_open, selected_open = imgui.menu_item("Open", "", False, True)
                if clicked_open:
                    file_names = plyer.filechooser.open_file()
                    if len(file_names) > 0:
                        data.open_image(file_names[0])
                        glfw.set_window_attrib(self.window, glfw.RESIZABLE, True)
                        glfw.set_window_aspect_ratio(self.window, data.width(), data.height())
                        glfw.set_window_attrib(self.window, glfw.RESIZABLE, False)
                        self.window_width, self.window_height = glfw.get_window_size(self.window)
                        self.image_renderer = _ImageRenderer(data.img)
                        self.image_renderer.set_mask_color(self.mask_color)

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True
                )

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        # Poll cursor position.
        cursor_pos = glfw.get_cursor_pos(self.window)
        if self.image_renderer is not None:
            imgui.begin("Controls", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            clicked, current = imgui.combo("Seg Method", data.segmentation_index, data.segmentation_options)
            if clicked and current != data.segmentation_index:
                data.segmentation_index = current
                self.image_renderer.update_mask(data.run_segmentation(erode_dilate_size=50))
            
            changed, self.mask_color = imgui.color_edit4("Background Color", *self.mask_color)
            if changed:
                self.image_renderer.set_mask_color(self.mask_color)

            # Cursor edit mode
            _, self.edit_mode_index = imgui.combo("Edit Mode", self.edit_mode_index, self.edit_mode_options)
            if self.edit_mode_index == 0: # Move mode
                self.brush_radius = -1
            elif self.edit_mode_index == 1: # Brush mode
                if self.brush_radius < 0:
                    self.brush_radius = 20.0
                _, self.brush_radius = imgui.slider_float("Brush Radius", self.brush_radius, 5.0, 100.0)

                # Query button state.
                io = self.impl.io
                if not io.want_capture_mouse and glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                    pos_x = int(cursor_pos[0] / self.window_width * data.width())
                    pos_y = int(cursor_pos[1] / self.window_height * data.height())
                    data.mark_unknown(pos_x, pos_y, int(self.brush_radius))
                    self.image_renderer.update_mask_sub_image(
                        0.5 * np.ones((int(self.brush_radius), int(self.brush_radius)), dtype=np.float32),
                        pos_x, pos_y)
            imgui.end()

        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        if self.image_renderer is not None:
            self.image_renderer.render((
                cursor_pos[0] / self.window_width,
                cursor_pos[1] / self.window_height,
                self.brush_radius))

        imgui.render()
        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)


    """
    Destroy and release graphics context.
    """
    def __del__(self) -> None:
        glfw.terminate()
    
