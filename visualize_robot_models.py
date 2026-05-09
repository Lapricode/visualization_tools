import numpy as np
import time
import os
import example_robot_data
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import trimesh
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel


# === 🔧 Set these ===
# urdf_path = os.getcwd() + "/robot_models/go2_description/urdf/go2.urdf"
urdf_path = os.getcwd() + "/robot_models/kuka_iiwa_description/urdf/iiwa7.urdf"
package_path = os.getcwd() + "/robot_models"  # For mesh resolution

# === 📦 Load robot model with visuals
model, _, visual_model = pin.buildModelsFromUrdf(urdf_path, package_dirs = [package_path])
data = model.createData()
q = pin.neutral(model)

# === 🔁 Forward kinematics
# pin.forwardKinematics(model, data, q)
# pin.updateFramePlacements(model, data, visual_model)

# === 🎥 MeshCat
vis = meshcat.Visualizer().open()

# for geom_obj in visual_model.geometryObjects:
#     mesh_path = geom_obj.meshPath
#     try:
#         mesh = trimesh.load(mesh_path)
#         if not isinstance(mesh, trimesh.Trimesh):
#             print(f"Skipping non-trimesh geometry: {mesh_path}")
#             continue
#         geometry = geom.TriangularMeshGeometry(mesh.vertices, mesh.faces)
#         vis[f"robot/{geom_obj.name}"].set_object(geometry, geom.MeshLambertMaterial())
#         # transform = (data.oMg[geom_obj.parentJoint] * geom_obj.placement).homogeneous
#         # vis[f"robot/{geom_obj.name}"].set_transform(transform)
#     except Exception as e:
#         print(f"Failed to load {mesh_path}: {e}")

for geom_obj in visual_model.geometryObjects:
    mesh_path = geom_obj.meshPath
    try:
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Trimesh):
            meshes = [(mesh, geom_obj.name)]
        elif isinstance(mesh, trimesh.Scene):
            meshes = []
            for name, geometry in mesh.geometry.items():
                meshes.append((geometry, f"{geom_obj.name}_{name}"))
        else:
            print(f"Unsupported mesh type in: {mesh_path}")
            continue
        for submesh, name in meshes:
            geometry = geom.TriangularMeshGeometry(submesh.vertices, submesh.faces)
            vis[f"robot/{name}"].set_object(geometry, geom.MeshLambertMaterial())
            # transform = (data.oMg[geom_obj.parentJoint] * geom_obj.placement).homogeneous
            # vis[f"robot/{name}"].set_transform(transform)
    except Exception as e:
        print(f"Failed to load {mesh_path}: {e}")


# # Load the go1 robot model
# robot = example_robot_data.load("double_pendulum")
# model = robot.model
# data = model.createData()

# # Compute forward kinematics
# q0 = robot.q0
# print(f"q0: {q0}")
# pin.forwardKinematics(model, data, q0)
# # pin.updateFramePlacements(model, data)

# # # Print visual geometry details to verify mesh files are defined
# # print("Visual geometries:")
# # for geom in robot.visual_model.geometryObjects:
# #     print(" -", geom.name, "mesh:", geom.meshPath)

# # Create a MeshCatVisualizer instance
# viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
# viz.initViewer(open = True)
# viz.loadViewerModel()
# viz.display(q0)  # Display the robot in its default configuration

# # q = q0.copy()
# # for i in range(1000):
# #     q[0] += 0.01
# #     viz.display(q)
# #     time.sleep(0.01)

# # Add a cube to the MeshCat environment
# cube_path = "environment/cube"
# cube_size = 1.0
# cube_color = [1.0, 0.0, 0.0, 1.0]  # RGBA
# viz.viewer[cube_path].set_object(
#     geom.Box([cube_size, cube_size, cube_size]),
#     geom.MeshLambertMaterial(color = 0x00ff00, transparent = True, opacity = 0.5, reflectivity = 0.5)  # or MeshPhongMaterial
# )
# cube_translation = [0.5, 0.5, cube_size / 2.0]
# cube_rotation = tf.identity_matrix()
# cube_transform = tf.translation_matrix(cube_translation) @ cube_rotation
# viz.viewer[cube_path].set_transform(cube_transform)


# robot_model = RobotModel(robot_name = 'franka')
# print(f"Absolute urdffile: {robot_model.get_urdf_path()}")
# print(f"Absolute xmlfile: {robot_model.get_xml_path()}")
# urdf_model = robot_model.yourdf_model()
# home_cfg = robot_model.home_cfg()
# urdf_model.update_cfg(configuration = home_cfg)
# urdf_model.show()
# # robot_model.copy_model(path = 'path/to/your/local/folder')
# # robot_model = LocalRobotModel(robot_name = 'name_folder')
