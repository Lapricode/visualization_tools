import time
from dataclasses import dataclass

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as np


@dataclass
class TerrainParams:
    grid_side: float = 10.0
    grid_res: int = 100
    half_1_color: int = 0xff0000
    half_2_color: int = 0x333333
    opacity: float = 0.5
    reflectivity: float = 0.5
    arrow_stride: int = 500
    arrow_length: float = 0.3
    arrow_radius: float = 0.01


def make_height_field(params: TerrainParams, height_fn):
    x = np.linspace(-params.grid_side, params.grid_side, params.grid_res)
    y = np.linspace(-params.grid_side, params.grid_side, params.grid_res)
    X, Y = np.meshgrid(x, y)  # same as your original code
    Z = height_fn(X, Y)
    vertices = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
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


def face_centroid_and_frame(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    e1 = p2 - p1
    e2 = p3 - p1

    centroid = (p1 + p2 + p3) / 3.0
    normal = np.cross(e1, e2)
    normal /= np.linalg.norm(normal) + 1e-7

    # Keep the same tangent logic style as your original code
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


def terrain_scene(height_fn, params: TerrainParams):
    viz = meshcat.visualizer.Visualizer()

    x, y, X, Y, Z, vertices = make_height_field(params, height_fn)
    nx, ny = len(x), len(y)
    faces_set1, faces_set2 = make_faces(nx, ny)

    add_mesh(viz, "environment/grid/half_1", vertices, faces_set1,
             params.half_1_color, params.opacity, params.reflectivity)
    add_mesh(viz, "environment/grid/half_2", vertices, faces_set2,
             params.half_2_color, params.opacity, params.reflectivity)

    all_faces = np.vstack([faces_set1, faces_set2])
    for k, (i1, i2, i3) in enumerate(all_faces):
        if k % params.arrow_stride != 0:
            continue

        p1, p2, p3 = vertices[i1], vertices[i2], vertices[i3]
        centroid, rot_final, normal = face_centroid_and_frame(p1, p2, p3)

        # Same arrow placement logic as your original code
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
    params = TerrainParams()

    def height_fn(X, Y):
        return np.e ** (-X**2 - Y**2)

    viz = terrain_scene(height_fn, params)

    print("Keep this process running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
