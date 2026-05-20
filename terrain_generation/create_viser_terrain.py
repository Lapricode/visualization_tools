from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import time

import viser


@dataclass
class SceneConfig:
    use_custom_camera: bool = True
    camera_position: tuple[float, float, float] = (-10.0, 0.0, 7.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)

    ambient_color: tuple[float, float, float] = (0.8, 0.8, 1.0)
    ambient_intensity: float = 0.7

    directional_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    directional_intensity: float = 2.0
    directional_position: tuple[float, float, float] = (5.0, -5.0, 10.0)
    directional_cast_shadow: bool = True

    use_custom_lights: bool = True
    up_direction: str = "+z"


@dataclass
class TerrainParams:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    grid_res: int = 100
    z_scale: float = 1.0
    base_height: float = 0.0
    half_1_color: tuple[float, float, float] = (0.47, 0.47, 1.0)
    half_2_color: tuple[float, float, float] = (0.47, 1.0, 0.47)
    opacity: float = 1.0
    draw_frames: bool = True
    draw_only_normals: bool = True
    arrow_stride: int = 100
    arrow_length: float = 0.3

    @property
    def x_radius(self) -> float:
        return 0.5 * (self.x_max - self.x_min)

    @property
    def y_radius(self) -> float:
        return 0.5 * (self.y_max - self.y_min)

    @property
    def x_center(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def y_center(self) -> float:
        return 0.5 * (self.y_min + self.y_max)


def float_rgb_to_uint8(c: tuple[float, float, float]) -> tuple[int, int, int]:
    return (int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255)))


def normalize01(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    zmin, zmax = float(z.min()), float(z.max())
    if np.isclose(zmax, zmin):
        return np.zeros_like(z, dtype=np.float64)
    return (z - zmin) / (zmax - zmin)


def make_height_field(params: TerrainParams, height_fn):
    x = np.linspace(params.x_min, params.x_max, params.grid_res)
    y = np.linspace(params.y_min, params.y_max, params.grid_res)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z_raw = height_fn(X, Y)
    Z = params.base_height + params.z_scale * normalize01(Z_raw)
    vertices = np.c_[X.ravel(order="C"), Y.ravel(order="C"), Z.ravel(order="C")]
    return x, y, X, Y, Z, vertices


def make_faces(nx: int, ny: int):
    faces_set1, faces_set2 = [], []
    for i in range(nx - 1):
        for j in range(ny - 1):
            idx = j * nx + i
            faces_set1.append([idx, idx + 1, idx + nx])
            faces_set2.append([idx + 1, idx + nx + 1, idx + nx])
    return (
        np.asarray(faces_set1, dtype=np.uint32),
        np.asarray(faces_set2, dtype=np.uint32),
    )


def face_centroid_and_frame(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    e1 = p2 - p1
    e2 = p3 - p1
    centroid = (p1 + p2 + p3) / 3.0
    normal = np.cross(e1, e2)
    normal = normal / (np.linalg.norm(normal) + 1e-7)
    if abs(e2[1]) > 1e-12:
        xtangent = e1 - (e1[1] / e2[1]) * e2
    else:
        xtangent = e2.copy()
    xtangent = xtangent / (np.linalg.norm(xtangent) + 1e-7)
    ytangent = np.cross(normal, xtangent)
    return centroid, normal, xtangent, ytangent


def build_arrow_segments(all_faces: np.ndarray, vertices: np.ndarray, params: TerrainParams):
    normals_segs, x_segs, y_segs = [], [], []
    for k, (i1, i2, i3) in enumerate(all_faces):
        if k % params.arrow_stride != 0:
            continue
        p1, p2, p3 = vertices[i1], vertices[i2], vertices[i3]
        centroid, normal, xtangent, ytangent = face_centroid_and_frame(p1, p2, p3)
        normals_segs.append([centroid, centroid + params.arrow_length * normal])
        if not params.draw_only_normals:
            x_segs.append([centroid, centroid + params.arrow_length * xtangent])
            y_segs.append([centroid, centroid + params.arrow_length * ytangent])
    to_arr = lambda lst: np.asarray(lst, dtype=np.float32) if lst else np.empty((0, 2, 3), dtype=np.float32)
    return to_arr(normals_segs), to_arr(x_segs), to_arr(y_segs)


def set_camera(server: viser.ViserServer, scene: SceneConfig) -> None:
    if not scene.use_custom_camera:
        return
    server.initial_camera.position = scene.camera_position
    server.initial_camera.look_at = scene.camera_target


def set_lighting(server: viser.ViserServer, cfg: SceneConfig) -> None:
    server.scene.set_up_direction(cfg.up_direction)

    if not cfg.use_custom_lights:
        return

    # Optional: keep or remove these depending on whether you want only custom lights.
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


def terrain_scene(height_fn, params: TerrainParams, scene: SceneConfig):
    server = viser.ViserServer()

    set_camera(server, scene)
    set_lighting(server, scene)

    x, y, X, Y, Z, vertices = make_height_field(params, height_fn)
    nx, ny = len(x), len(y)
    faces1, faces2 = make_faces(nx, ny)

    vertices_f32 = vertices.astype(np.float32)

    server.scene.add_mesh_simple(
        name="/terrain/half_1",
        vertices=vertices_f32,
        faces=faces1,
        color=params.half_1_color,
        opacity=params.opacity,
        side="double",
        material="standard",
        flat_shading=False,
        cast_shadow=True,
        receive_shadow=True,
    )
    server.scene.add_mesh_simple(
        name="/terrain/half_2",
        vertices=vertices_f32,
        faces=faces2,
        color=params.half_2_color,
        opacity=params.opacity,
        side="double",
        material="standard",
        flat_shading=False,
        cast_shadow=True,
        receive_shadow=True,
    )

    if params.draw_frames:
        all_faces = np.vstack([faces1, faces2])
        normals_segs, x_segs, y_segs = build_arrow_segments(all_faces, vertices, params)
        
        if normals_segs.shape[0] > 0:
            server.scene.add_line_segments(
                name="/frames/normals",
                points=normals_segs,
                colors=(0, 0, 255),
                line_width=2.0,
            )
        if not params.draw_only_normals:
            if x_segs.shape[0] > 0:
                server.scene.add_line_segments(
                    name="/frames/x_tangents",
                    points=x_segs,
                    colors=(255, 0, 0),
                    line_width=2.0,
                )
        
        if y_segs.shape[0] > 0:
            server.scene.add_line_segments(
                name="/frames/y_tangents",
                points=y_segs,
                colors=(0, 200, 0),
                line_width=2.0,
            )
    
    return server


if __name__ == "__main__":
    scene = SceneConfig(
        camera_position=(-10.0, 0.0, 7.0),
        camera_target=(0.0, 0.0, 0.0),
        # Ambient: dim, warm white — like MuJoCo's headlight_ambient
        ambient_color=(0.8, 0.8, 1.0),
        ambient_intensity=1.0,
        # Directional: strong, from above-left — like MuJoCo's light_pos
        directional_color=(1.0, 1.0, 1.0),
        directional_intensity=2.0,
        directional_position=(5.0, -5.0, 10.0),
        directional_cast_shadow=False,
        use_custom_lights=True,
    )

    params = TerrainParams(
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        grid_res=100,
        z_scale=1.0,
        base_height=0.1,
        half_1_color=(0.5, 1.0, 0.5),
        half_2_color=(1.0, 0.5, 0.5),
        opacity=0.7,
        draw_frames=True,
        draw_only_normals=False,
        arrow_stride=100,
        arrow_length=0.3,
    )

    def height_fn(X, Y):
        return (X**2 + Y**2) * np.exp(1.0 - (1 / 9) * (X**2 + Y**2))

    # Alternative height functions:
    # def height_fn(X, Y):
    #     return (X**2 + Y**2) * np.sin(X) * np.sin(Y) * np.exp(1.0 - (X**2 + Y**2))
    #
    # rng = np.random.default_rng(0)
    # def height_fn(X, Y):
    #     return rng.random(X.shape) * rng.random(X.shape)

    server = terrain_scene(height_fn, params, scene)

    print(f"Viser running at: http://localhost:{server.get_port()}")
    print("Open the URL above in your browser. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
