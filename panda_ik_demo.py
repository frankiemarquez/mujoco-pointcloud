"""
panda_ik_demo.py

Differential inverse kinematics demo, following the same pattern as the
manipulation tutorial this project is based on: a circular target
trajectory, tracked via a Jacobian-based differential IK step (same math:
mj_jacSite -> pseudo-inverse -> mj_integratePos), with the target and
end-effector paths drawn as trace capsules directly in the mjvScene -- all
lifted close to verbatim from that tutorial's own helper functions
(`circle`, `add_visual_capsule`, `modify_scene`).

Adaptations needed for OUR merged scene (the tutorial's original code
assumes a single-robot model where every qpos/qvel entry belongs to the
arm; ours also has 6 free objects on the table):
  - `data.ctrl = q` (tutorial) -> `data.ctrl[:7] = q[:7]` (only the arm's
    own 7 joint actuators; actuator8 is the gripper and is left alone).
  - `np.clip(q, *model.jnt_range.T, out=q)` (tutorial) -> clip only q[:7]
    against the arm's own joint ranges; the object freejoints don't have a
    1:1 qpos-to-range mapping the way single-DOF joints do.
  - Uses `scene_setup.reset_to_home` instead of a keyframe reset (see that
    module's docstring for why: this model's keyframe was removed because
    resetting via keyframe silently corrupts free-object placement once
    merged into a larger scene).

Run:
    MUJOCO_GL=egl python3 panda_ik_demo.py
Writes panda_ik_demo.mp4 (via mediapy, same video-writing library the
tutorial uses for `media.show_video` -- here `media.write_video` since
there's no notebook to display inline into).
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
import mediapy as media

from scene_setup import reset_to_home

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
reset_to_home(model, data, settle_steps=300)

renderer = mujoco.Renderer(model, height=360, width=480)
scene_option = mujoco.MjvOption()
scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE
scene_option.sitegroup[1] = 1  # make attachment_site / ik_target sites visible

mocap_id = model.body("ik_target").mocapid[0]
site_id = model.site("attachment_site").id

# ---- Target trajectory (identical pattern to the tutorial) ----------------
r = 0.12       # circle radius
cx, cy = 0.5, 0.0  # circle center (x, y)
cz = 0.55          # fixed height
f = 0.3            # trajectory frequency (Hz)


def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at
    (h, k), as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    return np.array([x, y])


# ---- Visualization helpers (same as the tutorial's) ------------------------
def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene. Same idea as the tutorial's helper;
    the underlying MuJoCo call was renamed `mjv_makeConnector` ->
    `mjv_connector` (and now takes point1/point2 as arrays, not 6 separate
    floats) in the installed mujoco version here."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                         mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                         np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom - 1],
                         mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         point1, point2)


def modify_scene(scn, target_traj, end_effector_traj):
    """Draw position traces for target (blue) and end-effector (red)."""
    if len(target_traj) > 1:
        for i in range(len(target_traj) - 1):
            add_visual_capsule(scn, target_traj[i], target_traj[i + 1], 0.004,
                                np.array([0, 0, 1.0, 1.0]))
            add_visual_capsule(scn, end_effector_traj[i], end_effector_traj[i + 1], 0.004,
                                np.array([1.0, 0, 0, 0.8]))


# ---- Differential IK loop (same math as the tutorial) ----------------------
# ---- Differential IK loop (same math as the tutorial) ----------------------
# NOTE: duration/framerate kept modest on purpose. The Panda's real mesh
# geometry (thousands of triangles per link) makes each render call ~0.9s
# here -- far more than our old primitive-geom arm -- so this renders far
# fewer frames than the tutorial's own examples to keep total runtime sane.
# Physics itself still steps at the model's normal 2ms timestep throughout.
duration = 5    # (seconds)
framerate = 8   # (Hz)

frames = []
end_effector_traj = []
target_traj = []

jac = np.zeros((6, model.nv))
error = np.zeros(6)
error_pos = error[:3]
error_ori = error[3:]
site_quat = np.zeros(4)
target_quat_conj = np.zeros(4)
error_quat = np.zeros(4)

# Fix the target's orientation (identity) and only drive its position along
# the circle -- simpler than the tutorial's wrist-tracking case, appropriate
# for a fixed top-down attachment_site orientation.
data.mocap_quat[mocap_id] = np.array([1, 0, 0, 0])

t0 = data.time
while data.time - t0 < duration:
    t = data.time - t0
    xy = circle(t, r, cx, cy, f)
    data.mocap_pos[mocap_id, 0:2] = xy
    data.mocap_pos[mocap_id, 2] = cz

    # Position error in the world frame.
    error_pos[:] = data.site(site_id).xpos - data.mocap_pos[mocap_id]

    # Orientation error (axis-angle), same quaternion algebra as the tutorial.
    target_ori = data.mocap_quat[mocap_id]
    mujoco.mju_negQuat(target_quat_conj, target_ori)
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_mulQuat(error_quat, site_quat, target_quat_conj)
    mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

    # Jacobian of the attachment site, w.r.t. the FULL state (nv=45: 7 arm +
    # 2 finger + 6*6 object dofs). Only the arm's own columns are non-zero.
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    dq = np.linalg.pinv(jac) @ -error

    q = data.qpos.copy()
    mujoco.mj_integratePos(model, q, dq, 1)

    # Adaptation vs. the tutorial: clip + command ONLY the arm's own 7
    # joints -- q/jnt_range cover the whole merged model (51 qpos, 15
    # joints), not just the arm.
    np.clip(q[:7], *model.jnt_range[:7].T, out=q[:7])
    data.ctrl[:7] = q[:7]
    # Leave the gripper actuator (index 7) at whatever reset_to_home set.

    mujoco.mj_step(model, data)

    target_traj.append(data.mocap_pos[mocap_id].copy())
    end_effector_traj.append(data.site(site_id).xpos.copy())

    if len(frames) < (data.time - t0) * framerate:
        renderer.update_scene(data, camera="scene_cam", scene_option=scene_option)
        modify_scene(renderer.scene, target_traj[::5], end_effector_traj[::5])
        frames.append(renderer.render())

out_path = os.path.join(os.path.dirname(__file__), "panda_ik_demo.mp4")
media.write_video(out_path, frames, fps=framerate)
print(f"wrote {out_path} ({len(frames)} frames)")

final_err = np.linalg.norm(end_effector_traj[-1][:2] - target_traj[-1][:2])
print(f"final XY tracking error: {final_err*1000:.1f} mm")
