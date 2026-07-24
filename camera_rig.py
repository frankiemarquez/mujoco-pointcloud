"""
camera_rig.py

A free-flying camera rig, driven by keyboard input, implemented as a MuJoCo
mocap body (`cam_rig` in scene.xml) that carries the `scene_cam` camera.

Since it's a mocap body, its pose is NOT simulated by physics -- we write
directly to `data.mocap_pos[idx]` / `data.mocap_quat[idx]` every frame based
on accumulated keyboard input. This gives instant, physics-free camera
control while keeping the rest of the scene fully simulated.

Controls (see README.md):
  W/S       move forward / backward   (camera-local -Z / +Z)
  A/D       strafe left / right       (camera-local -X / +X)
  R/F       move up / down            (world Z)
  Arrow keys  yaw / pitch look
  Q/E       roll left / right
  Space     capture + push a fresh point cloud to the Open3D window
  C         toggle continuous (live) point-cloud streaming
  ESC       quit
"""
from __future__ import annotations
import numpy as np
import mujoco


def quat_from_euler_zyx(yaw, pitch, roll):
    """Return a mujoco-order quaternion (w,x,y,z) from yaw/pitch/roll (rad),
    applied intrinsically as R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)

    # quaternion multiplication qz * qy * qx
    def qmul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    qz = np.array([cy, 0, 0, sy])
    qy = np.array([cp, 0, sp, 0])
    qx = np.array([cr, sr, 0, 0])
    return qmul(qmul(qz, qy), qx)


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1 - 2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x*x+y*y)],
    ])


class CameraRig:
    """Free camera with position + yaw/pitch/roll, written into a mocap body."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 mocap_body_name: str = "cam_rig",
                 move_speed: float = 1.2, look_speed: float = 1.6):
        self.model = model
        self.data = data
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
        self.mocap_id = model.body_mocapid[body_id]
        if self.mocap_id < 0:
            raise ValueError(f"Body '{mocap_body_name}' is not a mocap body")

        self.pos = np.array(data.mocap_pos[self.mocap_id], dtype=float)
        q0 = np.array(data.mocap_quat[self.mocap_id], dtype=float)
        # decompose initial quat back to yaw/pitch/roll isn't unique/needed;
        # we just keep our own yaw/pitch/roll state starting at 0 relative
        # rotation applied on top of the current mocap orientation basis.
        self.base_R = quat_to_rotmat(q0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.move_speed = move_speed
        self.look_speed = look_speed

        # held-key state, updated by the viewer's key_callback
        self._keys_down = set()

    def key_down(self, key: str):
        self._keys_down.add(key)

    def key_up(self, key: str):
        self._keys_down.discard(key)

    def current_rotation(self):
        """World-frame rotation matrix (columns = camera x,y,z axes)."""
        dR = quat_to_rotmat(quat_from_euler_zyx(self.yaw, self.pitch, self.roll))
        return self.base_R @ dR

    def step(self, dt: float):
        """Integrate held keys into position/orientation and push to mocap."""
        R = self.current_rotation()
        x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]  # camera right/up/forward(-z)

        move = np.zeros(3)
        k = self._keys_down
        if "W" in k: move -= z_axis   # forward = camera -Z
        if "S" in k: move += z_axis
        if "A" in k: move -= x_axis
        if "D" in k: move += x_axis
        if "R" in k: move += np.array([0, 0, 1.0])
        if "F" in k: move -= np.array([0, 0, 1.0])

        if np.linalg.norm(move) > 0:
            move = move / np.linalg.norm(move)
        self.pos += move * self.move_speed * dt

        if "LEFT" in k: self.yaw += self.look_speed * dt
        if "RIGHT" in k: self.yaw -= self.look_speed * dt
        if "UP" in k: self.pitch += self.look_speed * dt
        if "DOWN" in k: self.pitch -= self.look_speed * dt
        if "Q" in k: self.roll += self.look_speed * dt
        if "E" in k: self.roll -= self.look_speed * dt

        self.pitch = np.clip(self.pitch, -1.5, 1.5)

        R = self.current_rotation()
        q = self._rotmat_to_quat(R)

        self.data.mocap_pos[self.mocap_id] = self.pos
        self.data.mocap_quat[self.mocap_id] = q

    @staticmethod
    def _rotmat_to_quat(m):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (m[2, 1] - m[1, 2]) / S
            y = (m[0, 2] - m[2, 0]) / S
            z = (m[1, 0] - m[0, 1]) / S
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)
