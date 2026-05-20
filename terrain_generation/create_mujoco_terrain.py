from __future__ import annotations
from dataclasses import dataclass

from pathlib import Path
import struct
import numpy as np

import mujoco
import mujoco.viewer


@dataclass
class MuJoCoScene:
    timestep: float = 0.001
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    integrator: str = "RK4"
    headlight_ambient: tuple[float, float, float] = (0.1, 0.1, 0.1)
    headlight_diffuse: tuple[float, float, float] = (0.3, 0.3, 0.3)
    headlight_specular: tuple[float, float, float] = (0.1, 0.1, 0.1)
    light_pos: tuple[float, float, float] = (0.0, 0.0, 20.0)
    camera_name: str = "main"
    camera_pos: tuple[float, float, float] = (0.0, -10.0, 10.0)
    camera_xyaxes: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 1.0)


@dataclass
class MuJoCoTerrainConfig:
    x_radius: float = 10.0
    y_radius: float = 10.0
    x_center: float = 0.0
    y_center: float = 0.0
    z_scale: float = 1.0
    base_height: float = 0.1
    friction: tuple[float, float, float] = (1.0, 0.1, 0.1)
    rgba: tuple[float, float, float, float] = (1.0, 1,0, 1.0, 1.0)


@dataclass
class BallConfig:
    start_pos: tuple[float, float, float] = (0.0, 0.0, 10.0)
    radius: float = 0.1
    density: float = 1.0
    friction: tuple[float, float, float] = (1.0, 0.1, 0.1)
    rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def normalize01(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype = np.float32)
    zmin = float(z.min())
    zmax = float(z.max())
    if np.isclose(zmax, zmin):
        return np.zeros_like(z, dtype = np.float32)
    return (z - zmin) / (zmax - zmin)


def write_hfield_binary(path: str | Path, z_norm01: np.ndarray):
    """
    MuJoCo custom hfield binary:
      int32 nrow
      int32 ncol
      float32 data[nrow*ncol] (row-major)
    """
    z_norm01 = np.asarray(z_norm01, dtype = np.float32)
    nrow, ncol = z_norm01.shape

    path = Path(path)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", nrow, ncol))
        f.write(z_norm01.tobytes(order = "C"))


def make_mjcf_with_hfield_and_ball(
    hfield_filename: str,
    scene_cfg: MuJoCoScene,
    terrain_cfg: MuJoCoTerrainConfig,
    ball_cfg: BallConfig,
) -> str:

    return f"""
<mujoco model="terrain_ball">

  <compiler angle="radian"/>

  <option
      timestep="{scene_cfg.timestep}"
      gravity="{scene_cfg.gravity[0]} {scene_cfg.gravity[1]} {scene_cfg.gravity[2]}"
      integrator="{scene_cfg.integrator}"/>

  <visual>
    <headlight
        diffuse="{scene_cfg.headlight_diffuse[0]} {scene_cfg.headlight_diffuse[1]} {scene_cfg.headlight_diffuse[2]}"
        ambient="{scene_cfg.headlight_ambient[0]} {scene_cfg.headlight_ambient[1]} {scene_cfg.headlight_ambient[2]}"
        specular="{scene_cfg.headlight_specular[0]} {scene_cfg.headlight_specular[1]} {scene_cfg.headlight_specular[2]}"/>
  </visual>

  <asset>
    <hfield
        name="terrain"
        file="{hfield_filename}"
        size="{terrain_cfg.x_radius}
              {terrain_cfg.y_radius}
              {terrain_cfg.z_scale}
              {terrain_cfg.base_height}"/>
  </asset>

  <worldbody>

    <geom
        name="ground"
        type="hfield"
        hfield="terrain"
        pos="{terrain_cfg.x_center} {terrain_cfg.y_center} 0"
        friction="{terrain_cfg.friction[0]} {terrain_cfg.friction[1]} {terrain_cfg.friction[2]}"
        rgba="{terrain_cfg.rgba[0]} {terrain_cfg.rgba[1]} {terrain_cfg.rgba[2]} {terrain_cfg.rgba[3]}"/>

    <body
        name="ball"
        pos="{ball_cfg.start_pos[0]} {ball_cfg.start_pos[1]} {ball_cfg.start_pos[2]}">

      <freejoint/>

      <geom
          name="ball_geom"
          type="sphere"
          size="{ball_cfg.radius}"
          density="{ball_cfg.density}"
          friction="{ball_cfg.friction[0]} {ball_cfg.friction[1]} {ball_cfg.friction[2]}"
          rgba="{ball_cfg.rgba[0]} {ball_cfg.rgba[1]} {ball_cfg.rgba[2]} {ball_cfg.rgba[3]}"/>

    </body>

    <light
        pos="{scene_cfg.light_pos[0]} {scene_cfg.light_pos[1]} {scene_cfg.light_pos[2]}"/>

    <camera
        name="{scene_cfg.camera_name}"
        pos="{scene_cfg.camera_pos[0]} {scene_cfg.camera_pos[1]} {scene_cfg.camera_pos[2]}"
        xyaxes="{scene_cfg.camera_xyaxes[0]} {scene_cfg.camera_xyaxes[1]} {scene_cfg.camera_xyaxes[2]}
                {scene_cfg.camera_xyaxes[3]} {scene_cfg.camera_xyaxes[4]} {scene_cfg.camera_xyaxes[5]}"/>

  </worldbody>

</mujoco>
"""


def build_terrain_files(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    resolution = 100
    nx, ny = resolution, resolution
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0
    base_height = 0.1
    z_scale = 1.0

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y, indexing = "xy")

    Z = (X**2 + Y**2) * np.exp(1.0 - (1/9) * (X**2 + Y**2))
    # Z = (X**2 + Y**2) * np.sin(X) * np.sin(Y) * np.exp(1.0 - (X**2 + Y**2))
    # Z = np.floor(X)
    # Z = np.random.rand(*X.shape) * np.random.rand(*Y.shape)
    Z_norm = normalize01(Z)
    hfield_path = output_dir / "terrain.hfield"
    xml_path = output_dir / "terrain.xml"

    write_hfield_binary(hfield_path, Z_norm)

    scene_cfg = MuJoCoScene(
        timestep = 0.001,
        gravity = (0.0, 0.0, -9.81),
        integrator = "RK4",
        light_pos = (0.0, 0.0, 20.0),
        camera_pos = (0.0, -10.0, 10.0),
    )

    terrain_cfg = MuJoCoTerrainConfig(
        x_radius = (x_max - x_min) / 2,
        y_radius = (y_max - y_min) / 2,
        x_center = (x_min + x_max) / 2,
        y_center = (y_min + y_max) / 2,
        z_scale = z_scale,
        base_height = base_height,
        friction = (1.0, 0.1, 0.1),
        rgba = (0.5, 0.5, 1.0, 1.0),
    )

    ball_cfg = BallConfig(
        start_pos = (0.0, 0.0, 10.0),
        radius = 0.1,
        density = 1.0,
        friction = (1.0, 0.1, 0.1),
        rgba = (0.5, 1.0, 0.5, 1.0),
    )

    xml_path.write_text(
        make_mjcf_with_hfield_and_ball(hfield_path.name, scene_cfg, terrain_cfg, ball_cfg),
        encoding = "utf-8",
    )

    return xml_path, hfield_path


if __name__ == "__main__":
    xml_path, hfield_path = build_terrain_files("mujoco_terrain")
    print("Wrote:", xml_path)
    print("Wrote:", hfield_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mujoco.viewer.launch(model, data)

    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     while viewer.is_running():
    #         mujoco.mj_step(model, data)
    #         viewer.sync()
