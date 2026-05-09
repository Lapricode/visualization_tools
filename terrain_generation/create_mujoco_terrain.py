from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np
import mujoco
import mujoco.viewer


@dataclass
class MuJoCoTerrainConfig:
    x_radius: float = 10.0
    y_radius: float = 10.0
    elevation_scale: float = 1.5
    base_thickness: float = 0.2
    rgba: tuple[float, float, float, float] = (0.7, 0.6, 0.5, 1.0)


@dataclass
class BallConfig:
    radius: float = 0.25
    density: float = 800.0
    rgba: tuple[float, float, float, float] = (0.2, 0.4, 0.9, 1.0)
    start_pos: tuple[float, float, float] = (0.0, 0.0, 3.0)


def normalize01(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    zmin = float(z.min())
    zmax = float(z.max())
    if np.isclose(zmax, zmin):
        return np.zeros_like(z, dtype=np.float32)
    return (z - zmin) / (zmax - zmin)


def write_hfield_binary(path: str | Path, elevation_01: np.ndarray):
    """
    MuJoCo custom hfield binary:
      int32 nrow
      int32 ncol
      float32 data[nrow*ncol]   (row-major)
    """
    elevation_01 = np.asarray(elevation_01, dtype=np.float32)
    nrow, ncol = elevation_01.shape

    path = Path(path)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", nrow, ncol))
        f.write(elevation_01.tobytes(order="C"))


def make_mjcf_with_hfield_and_ball(
    hfield_filename: str,
    terrain_cfg: MuJoCoTerrainConfig,
    ball_cfg: BallConfig,
) -> str:
    tr, tg, tb, ta = terrain_cfg.rgba
    br, bg, bb, ba = ball_cfg.rgba
    bx, by, bz = ball_cfg.start_pos

    return f"""
<mujoco model="terrain_ball">

  <compiler angle="radian"/>
  
  <option timestep="0.002"
          gravity="0 0 -9.81"
          integrator="RK4"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8"
               ambient="0.3 0.3 0.3"
               specular="0.1 0.1 0.1"/>
  </visual>

  <asset>
    <hfield name="terrain"
            file="{hfield_filename}"
            size="{terrain_cfg.x_radius}
                  {terrain_cfg.y_radius}
                  {terrain_cfg.elevation_scale}
                  {terrain_cfg.base_thickness}"/>
  </asset>

  <worldbody>

    <!-- Terrain -->
    <geom name="ground"
          type="hfield"
          hfield="terrain"
          rgba="{tr} {tg} {tb} {ta}"
          friction="1.0 0.1 0.1"/>

    <!-- Ball -->
    <body name="ball"
          pos="{bx} {by} {bz}">

      <freejoint/>

      <geom name="ball_geom"
            type="sphere"
            size="{ball_cfg.radius}"
            density="{ball_cfg.density}"
            rgba="{br} {bg} {bb} {ba}"
            friction="0.8 0.05 0.05"/>

    </body>

    <light pos="0 0 20"/>

    <camera name="main"
            pos="0 -18 10"
            xyaxes="1 0 0 0 0.5 1"/>

  </worldbody>

</mujoco>
"""


def build_terrain_files(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nx, ny = 200, 200
    x = np.linspace(-10.0, 10.0, nx)
    y = np.linspace(-10.0, 10.0, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    Z = np.exp(-(X**2 + Y**2))
    Z01 = normalize01(Z)

    hfield_path = output_dir / "terrain.hfield"
    xml_path = output_dir / "terrain.xml"

    write_hfield_binary(hfield_path, Z01)

    terrain_cfg = MuJoCoTerrainConfig(
        x_radius=10.0,
        y_radius=10.0,
        elevation_scale=1.5,
        base_thickness=0.2,
    )
    ball_cfg = BallConfig(
        radius=0.5,
        density=500.0,
        rgba=(0.1, 0.3, 1.0, 1.0),
        start_pos=(0.0, 0.0, 8.0),
    )

    xml_path.write_text(
        make_mjcf_with_hfield_and_ball(hfield_path.name, terrain_cfg, ball_cfg),
        encoding="utf-8",
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
