"""
arm_control.py

Keyboard-driven Cartesian control of the Panda's end effector, via
differential inverse kinematics -- the same core algorithm as
`panda_ik_demo.py` (and, upstream of that, Kevin Zakka's `mjctrl` repo's
`diffik.py`): position error -> `mj_jacSite` -> pseudo-inverse -> joint
velocity -> `mj_integratePos`. The difference here is *what drives the
target*: `panda_ik_demo.py` scripts a circle; this reads held keyboard
keys each frame and moves the target in Cartesian space accordingly, then
runs the same IK step to command the arm to follow it.

Mirrors `camera_rig.CameraRig`'s shape (a small stateful controller with
`key_down`/`key_up`/`step(dt)`), so it plugs into the same Open3D
`register_key_action_callback` pattern used for the camera in
`interactive_pointcloud.py`.
"""
from __future__ import annotations
import numpy as np
import mujoco


class ArmIKController:
    """Jogs `mocap_body_name`'s position via held keys, then drives the
    Panda's arm joints (via `data.ctrl[:7]`) to track it with differential
    IK. Orientation is held fixed (identity) -- this controls *where* the
    gripper is, not how it's rotated.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 mocap_body_name: str = "ik_target",
                 site_name: str = "attachment_site",
                 move_speed: float = 0.25,
                 damping: float = 0.08,
                 workspace_bounds=((0.2, 0.75), (-0.4, 0.4), (0.15, 0.85))):
        self.model = model
        self.data = data
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
        self.mocap_id = model.body_mocapid[body_id]
        if self.mocap_id < 0:
            raise ValueError(f"Body '{mocap_body_name}' is not a mocap body")
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        self.pos = np.array(data.mocap_pos[self.mocap_id], dtype=float)
        data.mocap_quat[self.mocap_id] = np.array([1, 0, 0, 0])  # fixed orientation

        self.move_speed = move_speed
        self.damping = damping
        self.bounds = workspace_bounds  # ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        self.n_arm_joints = 7  # Panda: actuator1..7 (actuator8 is the separate gripper)

        # Pre-allocated IK working arrays.
        self.jac = np.zeros((6, model.nv))
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]
        self.site_quat = np.zeros(4)
        self.target_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

        self._keys_down = set()

    def key_down(self, key: str):
        self._keys_down.add(key)

    def key_up(self, key: str):
        self._keys_down.discard(key)

    def step(self, dt: float):
        """Update the target position from held keys, then run one
        differential-IK step to move the arm toward it."""
        model, data = self.model, self.data
        k = self._keys_down

        move = np.zeros(3)
        if "I" in k: move[0] += 1  # +X: away from the robot base
        if "K" in k: move[0] -= 1  # -X: toward the robot base
        if "J" in k: move[1] -= 1  # -Y: left
        if "L" in k: move[1] += 1  # +Y: right
        if "U" in k: move[2] += 1  # +Z: up
        if "O" in k: move[2] -= 1  # -Z: down

        if np.linalg.norm(move) > 0:
            move = move / np.linalg.norm(move)
            self.pos += move * self.move_speed * dt
            for i in range(3):
                lo, hi = self.bounds[i]
                self.pos[i] = np.clip(self.pos[i], lo, hi)

        data.mocap_pos[self.mocap_id] = self.pos

        # ---- Differential IK step (same math as panda_ik_demo.py / mjctrl's diffik.py) ----
        self.error_pos[:] = data.site(self.site_id).xpos - data.mocap_pos[self.mocap_id]

        target_ori = data.mocap_quat[self.mocap_id]
        mujoco.mju_negQuat(self.target_quat_conj, target_ori)
        mujoco.mju_mat2Quat(self.site_quat, data.site(self.site_id).xmat)
        mujoco.mju_mulQuat(self.error_quat, self.site_quat, self.target_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        mujoco.mj_jacSite(model, data, self.jac[:3], self.jac[3:], self.site_id)

        # Damped least-squares (Buss 2009, the same reference mjctrl cites):
        # dq = J^T (J J^T + lambda^2 I)^-1 (-error). Far more robust near
        # singularities / workspace edges than the plain pseudo-inverse
        # (np.linalg.pinv) used in panda_ik_demo.py's gentler circle
        # trajectory -- keyboard jogging is more likely to push toward
        # joint/workspace limits, where plain pinv can blow up.
        JJt = self.jac @ self.jac.T
        lam_sq = self.damping ** 2
        dq = self.jac.T @ np.linalg.solve(JJt + lam_sq * np.eye(6), -self.error)

        q = data.qpos.copy()
        mujoco.mj_integratePos(model, q, dq, 1)

        n = self.n_arm_joints
        np.clip(q[:n], *model.jnt_range[:n].T, out=q[:n])
        data.ctrl[:n] = q[:n]
