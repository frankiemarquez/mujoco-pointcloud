"""
show_rgbd_pipeline.py

Visual proof that the RGB camera, depth camera, and colorized point-cloud
extraction actually work together: renders RGB + depth from the same
camera, saves both as images (depth shown with a colormap so you can
actually see it), and saves the resulting colorized point cloud as a .ply.

Run:
    MUJOCO_GL=egl python3 show_rgbd_pipeline.py     # Linux headless
    python3 show_rgbd_pipeline.py                    # macOS / most setups

Outputs (in ./proof/):
    01_rgb.png          -- what a normal color camera sees
    02_depth.png         -- depth image, colorized (yellow=near, purple=far)
    03_pointcloud.ply    -- the resulting colorized 3D point cloud
    04_side_by_side.png  -- RGB | depth | point-cloud-projection, for a
                            single screenshot you can drop in a report/slide
"""
import os
import numpy as np
import mujoco
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from pointcloud_utils import capture_pointcloud, render_rgbd
from scene_setup import reset_to_home, disable_shadows

OUT_DIR = os.path.join(os.path.dirname(__file__), "proof")
os.makedirs(OUT_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), "scene.xml"))
data = mujoco.MjData(model)
reset_to_home(model, data)  # Panda in 'home' pose, objects settled
disable_shadows(model)  # ~5x faster offscreen capture (see scene_setup.py)

renderer = mujoco.Renderer(model, height=480, width=640)

# 1) RGB + depth from the SAME camera
rgb, depth = render_rgbd(renderer, data, "scene_cam")
imageio.imwrite(os.path.join(OUT_DIR, "01_rgb.png"), rgb)
print(f"RGB image: {rgb.shape}, dtype={rgb.dtype}")
print(f"Depth image: {depth.shape}, dtype={depth.dtype}, "
      f"range=[{depth.min():.2f}, {depth.max():.2f}] meters")

# 2) Colorize the depth image so it's actually visible (raw depth is just
#    a grid of float meters -- not visually meaningful without a colormap)
depth_vis = np.clip(depth, 0, 4.0)  # clip background/sky to 4m for contrast
plt.imsave(os.path.join(OUT_DIR, "02_depth.png"), depth_vis, cmap="viridis")

# 3) The colorized point cloud itself
points, colors = capture_pointcloud(model, data, renderer, "scene_cam", depth_trunc=4.0)
print(f"Point cloud: {points.shape[0]} points, each with RGB color")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(OUT_DIR, "03_pointcloud.ply"), pcd)

# 4) One combined figure: RGB | depth | point cloud (2D projection, so it
#    renders even without a display) -- good for a report/slide/README
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb)
axes[0].set_title("1) RGB camera")
axes[0].axis("off")

im = axes[1].imshow(depth_vis, cmap="viridis")
axes[1].set_title("2) Depth camera (meters)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046, label="meters")

axes[2].scatter(points[:, 0], points[:, 1], c=colors, s=0.5)
axes[2].set_title("3) Colorized point cloud (top-down)")
axes[2].set_xlabel("X (m)")
axes[2].set_ylabel("Y (m)")
axes[2].axis("equal")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_side_by_side.png"), dpi=140)
print(f"\nWrote proof images + point cloud to: {OUT_DIR}/")
