"""
demo_flythrough.py

Headless (no display needed) proof-of-concept:
  - Simulates the scene for a bit so objects settle on the table.
  - Sweeps the point-cloud camera through several poses (as the interactive
    keyboard controller would move it).
  - At each pose, captures an RGB image + a colorized point cloud (.ply).
  - Also renders an Open3D-rasterized top-down view of the merged point
    cloud to `demo_merged_topdown.png` as a sanity check, entirely offscreen.

Run:  MUJOCO_GL=egl python3 demo_flythrough.py
"""
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
import imageio.v2 as imageio

from camera_rig import CameraRig
from pointcloud_utils import capture_pointcloud, render_rgbd
from scene_setup import reset_to_home, disable_shadows

OUT_DIR = os.path.join(os.path.dirname(__file__), "demo_output")
os.makedirs(OUT_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), "scene.xml"))
data = mujoco.MjData(model)
reset_to_home(model, data, settle_steps=500)
disable_shadows(model)  # ~5x faster offscreen capture (see scene_setup.py)

renderer = mujoco.Renderer(model, height=480, width=640)
rig = CameraRig(model, data)

# A scripted sequence of "keyboard" moves, exactly like a user tapping keys.
moves = (
    [("W", 40)] +
    [("LEFT", 30)] +
    [("W", 20)] +
    [("UP", 15)] +
    [("D", 20)]
)

all_points = []
all_colors = []

frame = 0
for key, n in moves:
    rig.key_down(key)
    for _ in range(n):
        rig.step(0.02)
        mujoco.mj_step(model, data)
    rig.key_up(key)

    rgb, _ = render_rgbd(renderer, data, "scene_cam")
    imageio.imwrite(os.path.join(OUT_DIR, f"frame_{frame:02d}_rgb.png"), rgb)

    points, colors = capture_pointcloud(model, data, renderer, "scene_cam", depth_trunc=4.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(OUT_DIR, f"frame_{frame:02d}_cloud.ply"), pcd)

    all_points.append(points)
    all_colors.append(colors)
    print(f"frame {frame}: cam_pos={rig.pos.round(3)}  n_points={len(points)}")
    frame += 1

merged_pts = np.concatenate(all_points, axis=0)
merged_cols = np.concatenate(all_colors, axis=0)
merged = o3d.geometry.PointCloud()
merged.points = o3d.utility.Vector3dVector(merged_pts)
merged.colors = o3d.utility.Vector3dVector(merged_cols)
merged = merged.voxel_down_sample(voxel_size=0.005)
o3d.io.write_point_cloud(os.path.join(OUT_DIR, "merged_cloud.ply"), merged)
print(f"merged cloud: {len(merged.points)} points (after voxel downsample) -> merged_cloud.ply")

# Lightweight sanity-check image (no GL context needed -- Open3D's own
# offscreen GL renderer needs a real GPU/EGL display stack that may not be
# present in a headless container; matplotlib gives us a portable check).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pts_np = np.asarray(merged.points)
cols_np = np.asarray(merged.colors)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(pts_np[:, 0], pts_np[:, 1], c=cols_np, s=0.5)
axes[0].set_title("Top-down (X-Y)")
axes[0].set_xlabel("X"); axes[0].set_ylabel("Y"); axes[0].axis("equal")
axes[1].scatter(pts_np[:, 0], pts_np[:, 2], c=cols_np, s=0.5)
axes[1].set_title("Side (X-Z)")
axes[1].set_xlabel("X"); axes[1].set_ylabel("Z"); axes[1].axis("equal")
plt.tight_layout()
out_png = os.path.join(OUT_DIR, "merged_cloud_projections.png")
plt.savefig(out_png, dpi=130)
print("wrote", out_png)
