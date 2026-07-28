"""Headless sanity test: render RGB/D from scene_cam, build a colorized
point cloud, and save both a PNG (for visual inspection) and a .ply file
(for opening in any 3D viewer, including Open3D)."""
import os
import sys
if sys.platform.startswith("linux"):
    # EGL is a Linux-only offscreen backend; on macOS/Windows, let mujoco
    # pick its own default (setting MUJOCO_GL=egl there raises at import
    # time) -- only needed for this project's original headless Linux
    # dev sandbox.
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import open3d as o3d
from pointcloud_utils import capture_pointcloud, render_rgbd
from scene_setup import reset_to_home

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Panda in its 'home' pose, objects settled onto the table under gravity.
reset_to_home(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)

rgb, depth = render_rgbd(renderer, data, "scene_cam")
print("rgb", rgb.shape, rgb.dtype, "depth", depth.shape, depth.dtype,
      "depth range", depth.min(), depth.max())

# Save the RGB frame as PNG for a quick visual check.
import imageio.v2 as imageio
imageio.imwrite(os.path.join(os.path.dirname(__file__), "test_rgb.png"), rgb)

points, colors = capture_pointcloud(model, data, renderer, "scene_cam", depth_trunc=4.0)
print("point cloud:", points.shape, colors.shape)
print("bounds min", points.min(axis=0), "max", points.max(axis=0))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
out_ply = os.path.join(os.path.dirname(__file__), "test_pointcloud.ply")
o3d.io.write_point_cloud(out_ply, pcd)
print("wrote", out_ply)
