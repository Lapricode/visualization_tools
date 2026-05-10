from __future__ import annotations

import time
from dataclasses import dataclass

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as np


@dataclass
class SceneConfig:
    camera_position: tuple[float, float, float] = (0.0, -10.0, 10.0)
    light_position: tuple[float, float, float] = (0.0, 0.0, 20.0)
    show_camera_marker: bool = True
    show_light_marker: bool = True
    camera_marker_color: int = 0x0000FF
    light_marker_color: int = 0xFFFF00
    marker_radius: float = 0.15


@dataclass
class TerrainParams:
    # Domain
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0

    # Grid
    grid_res: int = 100

    # Vertical mapping
    z_scale: float = 1.0
    base_height: float = 0.0

    # Surface appearance
    half_1_color: int = 0xff0000
    half_2_color: int = 0x333333
    opacity: float = 0.5
    reflectivity: float = 0.5

    # Normal/tangent arrows
    arrow_stride: int = 500
    arrow_length: float = 0.3
    arrow_radius: float = 0.01

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


def normalize01(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    zmin = float(z.min())
    zmax = float(z.max())
    if np.isclose(zmax, zmin):
        return np.zeros_like(z, dtype=np.float32)
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
    faces_set1 = []
    faces_set2 = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            idx = j * nx + i
            faces_set1.append([idx, idx + 1, idx + nx])
            faces_set2.append([idx + 1, idx + nx + 1, idx + nx])

    return np.asarray(faces_set1, dtype=np.int32), np.asarray(faces_set2, dtype=np.int32)


def add_mesh(viz, path: str, vertices: np.ndarray, faces: np.ndarray, color: int, opacity: float, reflectivity: float):
    viz[path].set_object(
        geom.TriangularMeshGeometry(vertices, faces),
        geom.MeshLambertMaterial(
            color=color,
            transparent=True,
            opacity=opacity,
            reflectivity=reflectivity,
        ),
    )


def add_marker(viz, path: str, position: tuple[float, float, float], color: int, radius: float):
    viz[path].set_object(
        geom.Sphere(radius),
        geom.MeshLambertMaterial(color=color),
    )
    T = np.eye(4)
    T[:3, 3] = np.asarray(position, dtype=float)
    viz[path].set_transform(T)


def face_centroid_and_frame(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    e1 = p2 - p1
    e2 = p3 - p1

    centroid = (p1 + p2 + p3) / 3.0
    normal = np.cross(e1, e2)
    normal /= np.linalg.norm(normal) + 1e-7

    if abs(e2[1]) > 1e-12:
        xtangent = e1 - (e1[1] / e2[1]) * e2
    else:
        xtangent = e2 - (e2[1] / (e1[1] + 1e-12)) * e1

    xtangent /= np.linalg.norm(xtangent) + 1e-7
    ytangent = np.cross(normal, xtangent)

    rot_final = np.column_stack([xtangent, ytangent, normal])
    return centroid, rot_final, normal


def add_arrow(viz, path: str, color: int, length: float, radius: float, transform: np.ndarray):
    viz[path].set_object(
        geom.Cylinder(radius=radius, height=length),
        geom.MeshLambertMaterial(color=color),
    )
    viz[path].set_transform(transform)


def terrain_scene(height_fn, params: TerrainParams, scene_cfg: SceneConfig):
    viz = meshcat.visualizer.Visualizer()

    x, y, X, Y, Z, vertices = make_height_field(params, height_fn)
    nx, ny = len(x), len(y)
    faces_set1, faces_set2 = make_faces(nx, ny)

    add_mesh(
        viz,
        "environment/grid/half_1",
        vertices,
        faces_set1,
        params.half_1_color,
        params.opacity,
        params.reflectivity,
    )
    add_mesh(
        viz,
        "environment/grid/half_2",
        vertices,
        faces_set2,
        params.half_2_color,
        params.opacity,
        params.reflectivity,
    )

    if scene_cfg.show_light_marker:
        add_marker(
            viz,
            "scene/light",
            scene_cfg.light_position,
            scene_cfg.light_marker_color,
            scene_cfg.marker_radius,
        )

    if scene_cfg.show_camera_marker:
        add_marker(
            viz,
            "scene/camera",
            scene_cfg.camera_position,
            scene_cfg.camera_marker_color,
            scene_cfg.marker_radius,
        )

    all_faces = np.vstack([faces_set1, faces_set2])
    for k, (i1, i2, i3) in enumerate(all_faces):
        if k % params.arrow_stride != 0:
            continue

        p1, p2, p3 = vertices[i1], vertices[i2], vertices[i3]
        centroid, rot_final, normal = face_centroid_and_frame(p1, p2, p3)

        T = np.eye(4)
        T[:3, :3] = rot_final
        T[:3, 3] = centroid

        transform_x = (
            tf.translation_matrix([params.arrow_length / 2.0, 0.0, 0.0])
            @ tf.rotation_matrix(-np.pi / 2, [0.0, 0.0, 1.0])
        )
        transform_y = tf.translation_matrix([0.0, params.arrow_length / 2.0, 0.0])
        transform_z = (
            tf.translation_matrix([0.0, 0.0, params.arrow_length / 2.0])
            @ tf.rotation_matrix(np.pi / 2, [1.0, 0.0, 0.0])
        )

        add_arrow(
            viz,
            f"xtangents/arrow_{k}",
            0xff0000,
            params.arrow_length,
            params.arrow_radius,
            T @ transform_x,
        )
        add_arrow(
            viz,
            f"ytangents/arrow_{k}",
            0x00ff00,
            params.arrow_length,
            params.arrow_radius,
            T @ transform_y,
        )
        add_arrow(
            viz,
            f"normals/arrow_{k}",
            0x0000ff,
            params.arrow_length,
            params.arrow_radius,
            T @ transform_z,
        )

    return viz


if __name__ == "__main__":
    scene_cfg = SceneConfig(
        camera_position=(0.0, -10.0, 10.0),
        light_position=(0.0, 0.0, 20.0),
    )

    params = TerrainParams(
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        grid_res=100,
        z_scale=0.1,
        base_height=0.1,
        half_1_color=0x8888FF,
        half_2_color=0x444444,
        opacity=0.8,
        reflectivity=0.3,
        arrow_stride=500,
        arrow_length=0.3,
        arrow_radius=0.01,
    )

    rng = np.random.default_rng(0)

    def height_fn(X, Y):
        return rng.random(X.shape) * rng.random(X.shape)

    viz = terrain_scene(height_fn, params, scene_cfg)

    print("Keep this process running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
