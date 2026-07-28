"""
arm_control.py

Keyboard-driven Cartesian control of the Panda's end effector, via
differential inverse kinematics WITH NULL-SPACE CONTROL -- the same idea as
Kevin Zakka's `mjctrl` repo's `diffik_nullspace.py` (as opposed to the
simpler `diffik.py`, which `panda_ik_demo.py` uses).

Why null-space control is needed here specifically: the Panda has 7 joints
but a Cartesian position target only constrains 3 (or 6 with orientation)
-- so infinitely many joint configurations put the end effector in the
same place. Plain differential IK (`panda_ik_demo.py`'s approach) has no
preference among them, so as you jog the target around with the keyboard,
the arm can drift into a valid-but-visually-awkward "elbow-flipped"
configuration. The fix: track the Cartesian target as the PRIMARY task,
and add a SECONDARY task -- pulling joints back toward the natural 'home'
configuration -- projected into the primary task's null space (i.e. only
acting in directions that don't interfere with tracking the target).

Math: dq = J^+ (-error) + (I - J^+ J) dq_null, where J^+ is the damped
pseudo-inverse and dq_null = Kn (q_home - q) biases toward home.

Mirrors `camera_rig.CameraRig`'s shape (a small stateful controller with
`key_down`/`key_up`/`step(dt)`), so it plugs into the same Open3D
`register_key_action_callback` pattern used for the camera in
`interactive_pointcloud.py`.
"""
from __future__ import annotations
import numpy as np
import mujoco

from scene_setup import PANDA_HOME_QPOS


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
                 nullspace_gain: float = 1.2,
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
        self.q_home = PANDA_HOME_QPOS[:7].copy()  # secondary task target
        # Per-joint null-space weight: wrist joints (5,6,7 -- indices 4,5,6)
        # visibly rotate the gripper, so they're pulled back toward home
        # more strongly than the shoulder/elbow joints (0,1,2 -- which need
        # more freedom to actually reach targets across the workspace).
        # This is what keeps the gripper from looking randomly twisted even
        # when the shoulder has to swing wide to reach a target.
        joint_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.3, 1.3, 1.3])
        self.nullspace_gain = nullspace_gain * joint_weights
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

        # Damped pseudo-inverse (Buss 2009, the same reference mjctrl
        # cites): J^+ = J^T (J J^T + lambda^2 I)^-1. Far more robust near
        # singularities / workspace edges than a plain pseudo-inverse --
        # keyboard jogging is more likely to push toward joint/workspace
        # limits than a gentle scripted trajectory.
        JJt = self.jac @ self.jac.T
        lam_sq = self.damping ** 2
        J_pinv = self.jac.T @ np.linalg.inv(JJt + lam_sq * np.eye(6))

        dq_task = J_pinv @ (-self.error)

        # Null-space secondary task (same idea as mjctrl's
        # diffik_nullspace.py): bias the arm's own 7 joints back toward
        # `q_home`, but ONLY in directions that don't disturb the primary
        # Cartesian task -- this is what keeps the elbow/wrist from
        # drifting into an ugly, valid-but-arbitrary configuration as you
        # jog the target around. Projector (I - J^+J) zeroes out any
        # component of dq_null that would move the end effector.
        dq_null = np.zeros(model.nv)
        dq_null[:self.n_arm_joints] = self.nullspace_gain * (self.q_home - data.qpos[:self.n_arm_joints])
        null_projector = np.eye(model.nv) - J_pinv @ self.jac
        dq = dq_task + null_projector @ dq_null

        q = data.qpos.copy()
        mujoco.mj_integratePos(model, q, dq, 1)

        n = self.n_arm_joints
        np.clip(q[:n], *model.jnt_range[:n].T, out=q[:n])
        data.ctrl[:n] = q[:n]
