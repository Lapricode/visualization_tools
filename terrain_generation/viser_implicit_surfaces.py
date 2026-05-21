from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from skimage import measure
import viser


@dataclass
class SceneConfig:
    use_custom_camera: bool = True
    camera_position: tuple[float, float, float] = (-4.0, -4.0, 2.5)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)

    ambient_color: tuple[float, float, float] = (0.8, 0.8, 1.0)
    ambient_intensity: float = 0.8

    directional_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    directional_intensity: float = 2.0
    directional_position: tuple[float, float, float] = (5.0, -5.0, 10.0)
    directional_cast_shadow: bool = True

    use_custom_lights: bool = True
    up_direction: str = "+z"


@dataclass
class ImplicitSurfaceParams:
    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -1.5
    y_max: float = 1.5
    z_min: float = -1.5
    z_max: float = 1.5
    grid_res: int = 80

    color: tuple[float, float, float] = (0.5, 0.8, 1.0)
    opacity: float = 0.95
    flat_shading: bool = False
    cast_shadow: bool = True
    receive_shadow: bool = True

    draw_axes: bool = True

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.x_min, self.y_min, self.z_min], dtype=np.float64)
        hi = np.array([self.x_max, self.y_max, self.z_max], dtype=np.float64)
        return lo, hi


def float_rgb_to_uint8(c: tuple[float, float, float]) -> tuple[int, int, int]:
    return (int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255)))


def set_camera(server: viser.ViserServer, scene: SceneConfig) -> None:
    if not scene.use_custom_camera:
        return
    server.initial_camera.position = scene.camera_position
    server.initial_camera.look_at = scene.camera_target


def set_lighting(server: viser.ViserServer, cfg: SceneConfig) -> None:
    server.scene.set_up_direction(cfg.up_direction)

    if not cfg.use_custom_lights:
        return

    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_ambient(
        name="/lights/ambient",
        color=float_rgb_to_uint8(cfg.ambient_color),
        intensity=cfg.ambient_intensity,
    )
    server.scene.add_light_directional(
        name="/lights/directional",
        color=float_rgb_to_uint8(cfg.directional_color),
        intensity=cfg.directional_intensity,
        position=cfg.directional_position,
        cast_shadow=cfg.directional_cast_shadow,
    )


def sphere_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, radius: float = 1.0) -> np.ndarray:
    return X**2 + Y**2 + Z**2 - radius**2


def torus_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, R: float = 0.9, r: float = 0.3) -> np.ndarray:
    q = np.sqrt(X**2 + Y**2) - R
    return q**2 + Z**2 - r**2


def gyroid_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return (
        np.sin(X) * np.cos(Y)
        + np.sin(Y) * np.cos(Z)
        + np.sin(Z) * np.cos(X)
    )


def make_implicit_mesh(params: ImplicitSurfaceParams, field_fn):
    xs = np.linspace(params.x_min, params.x_max, params.grid_res)
    ys = np.linspace(params.y_min, params.y_max, params.grid_res)
    zs = np.linspace(params.z_min, params.z_max, params.grid_res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    volume = field_fn(X, Y, Z)

    verts, faces, normals, _values = measure.marching_cubes(
        volume,
        level=0.0,
        spacing=(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]),
    )

    # marching_cubes returns coordinates relative to the grid origin; shift into world space.
    verts[:, 0] += params.x_min
    verts[:, 1] += params.y_min
    verts[:, 2] += params.z_min
    return verts.astype(np.float32), faces.astype(np.uint32), normals.astype(np.float32)


def implicit_surface_scene(field_fn, params: ImplicitSurfaceParams, scene: SceneConfig):
    server = viser.ViserServer()
    set_camera(server, scene)
    set_lighting(server, scene)

    verts, faces, _normals = make_implicit_mesh(params, field_fn)

    server.scene.add_mesh_simple(
        name="/implicit_surface",
        vertices=verts,
        faces=faces,
        color=params.color,
        opacity=params.opacity,
        side="double",
        material="standard",
        flat_shading=params.flat_shading,
        cast_shadow=params.cast_shadow,
        receive_shadow=params.receive_shadow,
    )

    if params.draw_axes:
        L = max(params.x_max - params.x_min, params.y_max - params.y_min, params.z_max - params.z_min) * 1.0
        server.scene.add_line_segments(
            name="/axes/x",
            points=np.array([[[0, 0, 0], [L, 0, 0]]], dtype=np.float32),
            colors=(255, 0, 0),
            line_width=3.0,
        )
        server.scene.add_line_segments(
            name="/axes/y",
            points=np.array([[[0, 0, 0], [0, L, 0]]], dtype=np.float32),
            colors=(0, 255, 0),
            line_width=3.0,
        )
        server.scene.add_line_segments(
            name="/axes/z",
            points=np.array([[[0, 0, 0], [0, 0, L]]], dtype=np.float32),
            colors=(0, 0, 255),
            line_width=3.0,
        )

    return server


if __name__ == "__main__":
    
    resolution = 100
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    z_min, z_max = -2.0, 2.0
    z_scale = 1.0
    base_height = 0.0

    scene = SceneConfig(
        camera_position=((x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2),
        camera_target=(0.0, 0.0, 0.0),
        ambient_intensity=0.9,
        directional_intensity=2.0,
        directional_position=(5.0, -5.0, 10.0),
        directional_cast_shadow=False,
        use_custom_lights=True,
    )

    params = ImplicitSurfaceParams(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        grid_res=resolution,
        color=(0.5, 0.5, 1.0),
        opacity=0.7,
    )

    # Swap this for torus_field or gyroid_field if you want a different implicit surface.
    field_fn = lambda X, Y, Z: sphere_field(X, Y, Z, radius = 1.0)
    # field_fn = lambda X, Y, Z: torus_field(X, Y, Z, R = 0.5, r = 0.25)
    # field_fn = lambda X, Y, Z: gyroid_field(X, Y, Z)

    server = implicit_surface_scene(field_fn, params, scene)

    print(f"Viser running at: http://localhost:{server.get_port()}")
    print("Open the URL above in your browser. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
