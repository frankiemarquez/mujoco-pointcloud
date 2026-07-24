"""
test_pointcloud_extraction.py

Focused proof that the colorized-point-cloud EXTRACTION INTERFACE itself
(`pointcloud_utils.capture_pointcloud`) works correctly -- as opposed to
`show_rgbd_pipeline.py`, which is about showing the RGB/depth cameras
visually. This script instead prints inspectable numeric evidence:

  1. Calls the interface exactly as documented (4 lines).
  2. Prints the shapes/dtypes of what comes back -- proving it really is
     an (N,3) XYZ array + an aligned (N,3) RGB array, not something else.
  3. Sanity-checks CORRECTNESS, not just shape: picks a pixel we know is
     looking at the red box and the green sphere, and verifies the
     extracted 3D point at roughly that location is actually colored red /
     green -- i.e. the interface didn't just invent plausible-looking
     numbers, it correctly fused depth + color per point.
  4. Saves the cloud as a .ply you can open in Open3D/MeshLab as further
     evidence.

Run:
    MUJOCO_GL=egl python3 test_pointcloud_extraction.py   # Linux headless
    python3 test_pointcloud_extraction.py                  # macOS
"""
import os
import numpy as np
import mujoco
import open3d as o3d

from pointcloud_utils import capture_pointcloud
from scene_setup import reset_to_home

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

# ---- 1) Use the interface exactly as documented ---------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
reset_to_home(model, data)  # Panda in 'home' pose, objects settled

renderer = mujoco.Renderer(model, height=480, width=640)

points, colors = capture_pointcloud(model, data, renderer, "scene_cam")

# ---- 2) Prove the shapes/dtypes are what the interface promises -----------
print("=" * 60)
print("INTERFACE OUTPUT CHECK")
print("=" * 60)
print(f"points: shape={points.shape}, dtype={points.dtype}")
print(f"colors: shape={colors.shape}, dtype={colors.dtype}")
assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
assert colors.ndim == 2 and colors.shape[1] == 3, "colors must be (N,3)"
assert points.shape[0] == colors.shape[0], "points and colors must align 1:1"
assert colors.min() >= 0.0 and colors.max() <= 1.0, "colors must be in [0,1]"
print("[PASS] points is (N,3) float XYZ, colors is (N,3) float RGB in [0,1], "
      "same N for both.")

# ---- 3) Prove CORRECTNESS: does color actually match known object colors? -
# Object positions come straight from the simulator's own body positions,
# not from our point-cloud code -- an independent ground truth to check against.
def body_world_pos(name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[body_id])

checks = [
    ("box_red",       "red",   0),  # index of the channel that should dominate
    ("sphere_green",  "green", 1),
    ("cylinder_blue", "blue",  2),
]

print("\n" + "=" * 60)
print("CORRECTNESS CHECK: does extracted color match the real object color?")
print("=" * 60)
print("(Comparing HUE -- which channel dominates -- not exact brightness,")
print(" since MuJoCo's lighting legitimately darkens/shades the flat")
print(" material color. A correct extraction should still show red points")
print(" as red-dominant, green points as green-dominant, etc.)\n")
for body_name, label, dominant_channel in checks:
    target = body_world_pos(body_name)
    dists = np.linalg.norm(points - target, axis=1)
    nearby = dists < 0.06  # points within 6cm of the object's center
    if nearby.sum() == 0:
        print(f"[WARN] no points found near '{body_name}' -- may be occluded "
              f"from this camera angle.")
        continue
    mean_color = colors[nearby].mean(axis=0)
    actual_dominant = int(np.argmax(mean_color))
    status = "PASS" if actual_dominant == dominant_channel else "FAIL"
    print(f"[{status}] '{body_name}': mean extracted RGB = {np.round(mean_color, 2)} "
          f"({nearby.sum()} points nearby) -> "
          f"{'RGB'[actual_dominant]}-dominant, expected {label}")

# ---- 4) Save as further evidence -------------------------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
out_path = os.path.join(os.path.dirname(__file__), "extraction_proof.ply")
o3d.io.write_point_cloud(out_path, pcd)
print(f"\nSaved point cloud to {out_path} -- open it in Open3D/MeshLab to "
      f"confirm visually too.")
