import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as np
import time


# Initialize the visualizer
viz = meshcat.visualizer.Visualizer()
print("URL:", viz.url())
viz.open()
viz.delete()  # Clear any existing objects
time.sleep(2.0)  # Give time for the visualizer to clear

# Add a cube to the MeshCat environment
cube_path = "environment/cube"
cube_size = 1.0
cube_color = 0x00ff00  # RGBA
viz[cube_path].set_object(
    geom.Box([cube_size, cube_size, cube_size]),
    geom.MeshLambertMaterial(color = cube_color, transparent = True, opacity = 0.5, reflectivity = 0.5)  # or MeshPhongMaterial
)
cube_translation = tf.translation_matrix([0.0, 0.0, cube_size / 2.0])
cube_rotation = tf.identity_matrix()
cube_transform = cube_translation @ cube_rotation
viz[cube_path].set_transform(cube_transform)

# Add a sphere to the MeshCat environment
sphere_path = "environment/sphere"
sphere_radius = 0.5
sphere_color = 0x0000ff
viz[sphere_path].set_object(
    geom.Sphere(radius = sphere_radius),
    geom.MeshLambertMaterial(color = sphere_color, transparent = True, opacity = 0.5, reflectivity = 0.5)  # or MeshPhongMaterial
)
sphere_translation = tf.translation_matrix([0.0, 0.0, 3. * sphere_radius])
sphere_rotation = tf.identity_matrix()
sphere_transform = sphere_translation @ sphere_rotation
viz[sphere_path].set_transform(sphere_transform)

# Add a cylinder to the MeshCat environment
cylinder_path = "environment/cylinder"
cylinder_radius = 0.2
cylinder_height = 2.0
cylinder_color = 0xffff00
viz[cylinder_path].set_object(
    geom.Cylinder(radius = cylinder_radius, height = cylinder_height),
    geom.MeshLambertMaterial(color = cylinder_color, transparent = True, opacity = 0.5, reflectivity = 0.5)  # or MeshPhongMaterial
)
cylinder_translation = tf.translation_matrix([0.0, 0.0, cylinder_height / 2.0])
cylinder_rotation = tf.rotation_matrix(np.pi / 2, [1.0, 0.0, 0.0])  # Rotate around the X-axis
cylinder_transform = cylinder_translation @ cylinder_rotation
viz[cylinder_path].set_transform(cylinder_transform)

# # Add a point cloud to the MeshCat environment
# point_cloud_path = "environment/point_cloud"
# point_cloud_size = 0.1
# point_cloud_color = np.array([1.0, 0.0, 1.0])  # RGB for magenta
# num_points = 100
# point_cloud_points = np.random.rand(num_points, 3) * 2 - 1  # Random points in [-1, 1]^3
# point_cloud_points[:, 2] = point_cloud_points[:, 2] * 0.5 + 0.5  # Adjust Z to be in [0, 1]
# viz[point_cloud_path].set_object(
#     geom.PointCloud(point_cloud_points, color = point_cloud_color, size = point_cloud_size),
# )
# point_cloud_translation = tf.translation_matrix([0.0, 0.0, 0.5])
# point_cloud_rotation = tf.identity_matrix()
# point_cloud_transform = point_cloud_translation @ point_cloud_rotation
# viz[point_cloud_path].set_transform(point_cloud_transform)

# Create a grid of points
grid_side = 10
grid_res = 100
x = np.linspace(-grid_side, grid_side, grid_res)
y = np.linspace(-grid_side, grid_side, grid_res)
X, Y = np.meshgrid(x, y)
# Z = np.zeros_like(X)
# Z = np.random.rand(*X.shape)
Z = np.sin(X) * np.cos(Y)
# Z = np.e**(-X**2 - Y**2)

nx = len(x)
ny = len(y)
faces_set1 = []
faces_set2 = []
vertices = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
for i in range(nx - 1):
    for j in range(ny - 1):
        idx = j * nx + i
        faces_set1.append([idx, idx + 1, idx + nx])
        faces_set2.append([idx + 1, idx + nx + 1, idx + nx])
        # print(f"{idx}:\t {[idx, idx + 1, idx + nx]},\t {vertices[idx]}, {vertices[idx + 1]}, {vertices[idx + nx]}")
faces_set1 = np.array(faces_set1)
faces_set2 = np.array(faces_set2)

viz["environment/grid/half_1"].set_object(
    geom.TriangularMeshGeometry(vertices, faces_set1),
    geom.MeshLambertMaterial(color = 0xff0000, transparent = True, opacity = 0.5, reflectivity = 0.5)
)
viz["environment/grid/half_2"].set_object(
    geom.TriangularMeshGeometry(vertices, faces_set2),
    geom.MeshLambertMaterial(color = 0x333333, transparent = True, opacity = 0.5, reflectivity = 0.5)
)

all_faces = np.vstack([faces_set1, faces_set2])
for k, (i1, i2, i3) in enumerate(all_faces):
    p1 = vertices[i1]
    p2 = vertices[i2]
    p3 = vertices[i3]

    e1 = p2 - p1
    e2 = p3 - p1

    centroid = (p1 + p2 + p3) / 3.0
    normal = np.cross(e1, e2)
    normal /= np.linalg.norm(normal) + 1e-7
    xtangent = (e1 - (e1[1] / e2[1]) * e2) if e2[1] != 0 else (e2 - (e2[1] / e1[1]) * e1)
    xtangent /= np.linalg.norm(xtangent) + 1e-7
    ytangent = np.cross(normal, xtangent)
    arrow_length = 0.3

    if k % 150 == 0:
        # Set arrow geometry
        viz[f"xtangents/arrow_{k}"].set_object(
            geom.Cylinder(radius = 0.01, height = arrow_length), geom.MeshLambertMaterial(color = 0xff0000)
        )
        viz[f"ytangents/arrow_{k}"].set_object(
            geom.Cylinder(radius = 0.01, height = arrow_length), geom.MeshLambertMaterial(color = 0x00ff00)
        )
        viz[f"normals/arrow_{k}"].set_object(
            geom.Cylinder(radius = 0.01, height = arrow_length), geom.MeshLambertMaterial(color = 0x0000ff)
        )

        # Build transformation matrix for arrow orientation
        rot_final = np.column_stack([xtangent, ytangent, normal])
        tranform_final = np.eye(4)
        tranform_final[:3, :3] = rot_final
        tranform_final[:3, 3] = centroid

        transform_x = tf.translation_matrix([arrow_length / 2.0, 0.0, 0.0]) @ tf.rotation_matrix(-np.pi / 2, [0.0, 0.0, 1.0])
        viz[f"xtangents/arrow_{k}"].set_transform(tranform_final @ transform_x)

        transform_y = tf.translation_matrix([0.0, arrow_length / 2.0, 0.0]) @ tf.identity_matrix()
        viz[f"ytangents/arrow_{k}"].set_transform(tranform_final @ transform_y)

        transform_z = tf.translation_matrix([0.0, 0.0, arrow_length / 2.0]) @ tf.rotation_matrix(np.pi / 2, [1.0, 0.0, 0.0])
        viz[f"normals/arrow_{k}"].set_transform(tranform_final @ transform_z)


# # Set the grid transform
# t_total = 10
# dt = 0.01
# omega = 2 * np.pi / t_total
# for k in range(int(t_total / dt) + 1):
#     t = k * dt
#     grid_translation = tf.translation_matrix([3. * np.sin(omega * t), 3. * np.cos(omega * t), 0.0])
#     grid_rotation = tf.identity_matrix()
#     # grid_rotation = tf.rotation_matrix((2 * np.pi) * (t + 1) / 100, [0.0, 0.0, 1.0])
#     grid_transform = grid_translation @ grid_rotation
#     viz["environment/grid"].set_transform(grid_transform)
#     time.sleep(dt)
