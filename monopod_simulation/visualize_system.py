#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


URDF_PATH = os.getcwd() + "/monopod_description/urdf/monopod.urdf"
MESH_DIR = os.getcwd() + "/monopod_description"

try:
    model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, [MESH_DIR], pin.JointModelFreeFlyer())
except Exception as e:
    print("Error: could not load URDF. Did you set URDF_PATH correctly?")
    print("Details:", e)
    sys.exit(1)

data = model.createData()
for geom in visual_model.geometryObjects:
    current_scale = geom.meshScale
    new_scale = tuple(s * 0.001 for s in current_scale)
    print(new_scale)
    # new_scale = current_scale
    geom.meshScale = new_scale

viz = MeshcatVisualizer(model, collision_model, visual_model)

viz.initViewer(open = True)
time.sleep(5)
viz.loadViewerModel()
viz.display(pin.neutral(model))
print("Model pushed. Open http://127.0.0.1:7000/ in your browser.")
