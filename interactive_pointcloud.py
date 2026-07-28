"""
interactive_pointcloud.py

Main interactive demo -- single window, cross-platform (Linux/macOS/Windows).

Design note (why there's no separate MuJoCo viewer window):
  An earlier version of this script also opened a native `mujoco.viewer`
  window alongside the Open3D window. On macOS that requires launching the
  whole script under `mjpython` (MuJoCo's viewer insists on owning the real
  main thread on macOS), and Open3D's window *also* wants the real main
  thread -- the two fight over it in a single process and crash. Since the
  actual requirement is "keyboard control + Open3D point-cloud view", this
  version drops the MuJoCo GUI window entirely and does everything --
  physics stepping, keyboard capture, and point-cloud updates -- inside
  Open3D's own event loop (`register_animation_callback` /
  `register_key_action_callback`). One window, one main thread, no
  mjpython needed, identical behavior on every platform.

Run:

    python3 interactive_pointcloud.py                  # macOS / most Linux setups
    MUJOCO_GL=egl python3 interactive_pointcloud.py    # Linux without a GLFW-capable X/Wayland session

Controls (Open3D window must have focus)
-----------------------------------------
  W / S            move camera forward / backward   (held)
  A / D            strafe camera left / right        (held)
  R / F            move camera up / down             (held)
  Left / Right     yaw camera                        (held)
  Up / Down        pitch camera                       (held)
  Q / E            roll camera                        (held)
  C                toggle continuous point-cloud streaming (on by default)
  Space            force a single point-cloud capture (useful if C is off)
  I / K            move arm end effector +X / -X      (held)
  J / L            move arm end effector -Y / +Y      (held)
  U / O            move arm end effector up / down     (held)
  N / M            open / close the gripper (Panda's tendon-coupled fingers)
  Esc              quit

Arm control is differential IK (see `arm_control.ArmIKController`, same
math as `panda_ik_demo.py`): I/J/K/L/U/O jog a Cartesian target the arm's
end effector then tracks every frame.
"""
import os
import time
import argparse

import numpy as np
import mujoco
import open3d as o3d

from camera_rig import CameraRig
from arm_control import ArmIKController
from pointcloud_utils import capture_pointcloud
from scene_setup import reset_to_home, PANDA_HOME_CTRL

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")


class GLFW:
    """Hardcoded GLFW constants (stable across GLFW versions). Avoids
    `import glfw`, which on macOS bundles its own libglfw.dylib and clashes
    with the one Open3D already links in-process (harmless but noisy
    'Class ... is implemented in both ...' warnings)."""
    PRESS, RELEASE, REPEAT = 1, 0, 2
    KEY_SPACE = 32
    KEY_A, KEY_C, KEY_D, KEY_E, KEY_F = 65, 67, 68, 69, 70
    KEY_I, KEY_J, KEY_K, KEY_L = 73, 74, 75, 76
    KEY_M, KEY_N, KEY_O, KEY_Q = 77, 78, 79, 81
    KEY_R, KEY_S, KEY_U, KEY_W = 82, 83, 85, 87
    KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265


glfw = GLFW  # keeps the rest of the file's `glfw.XXX` references unchanged

# GLFW key macros -> our camera rig's key names.
CAMERA_HELD_KEYS = {
    glfw.KEY_W: "W", glfw.KEY_S: "S", glfw.KEY_A: "A", glfw.KEY_D: "D",
    glfw.KEY_R: "R", glfw.KEY_F: "F",
    glfw.KEY_Q: "Q", glfw.KEY_E: "E",
    glfw.KEY_LEFT: "LEFT", glfw.KEY_RIGHT: "RIGHT",
    glfw.KEY_UP: "UP", glfw.KEY_DOWN: "DOWN",
}

# GLFW key macros -> our arm controller's key names.
ARM_HELD_KEYS = {
    glfw.KEY_I: "I", glfw.KEY_K: "K",
    glfw.KEY_J: "J", glfw.KEY_L: "L",
    glfw.KEY_U: "U", glfw.KEY_O: "O",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--depth_trunc", type=float, default=4.0)
    parser.add_argument("--stream_hz", type=float, default=12.0,
                         help="max rate (Hz) at which the point cloud is recomputed")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    reset_to_home(model, data)  # Panda in 'home' pose, objects settled onto table

    rig = CameraRig(model, data)
    arm = ArmIKController(model, data)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Gripper actuator (actuator8) uses ctrlrange 0-255 via a tendon coupling
    # both fingers -- NOT the same units as our old custom arm's two
    # separate 0-0.04m finger actuators. 255 = open, 0 = closed (see
    # scene_setup.py / franka_emika_panda/panda.xml's 'home' keyframe).
    state = {"continuous": True, "capture_once": True,
             "gripper": float(PANDA_HOME_CTRL[-1])}
    timing = {"t_prev": time.time(), "last_capture": 0.0}
    min_period = 1.0 / max(args.stream_hz, 0.1)

    # ---- Open3D window + geometry -------------------------------------
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Live Colorized Point Cloud (WASD/RF/QE/Arrows=camera, IJKLUO=arm, N/M=gripper)",
                       width=1000, height=750)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.08])
    opt.point_size = 2.5

    view_initialized = {"done": False}

    # ---- Held-key (press/repeat/release) callbacks ---------------------
    def make_hold_callback(controller, name):
        def cb(_vis, action, _mods):
            if action in (glfw.PRESS, glfw.REPEAT):
                controller.key_down(name)
            elif action == glfw.RELEASE:
                controller.key_up(name)
            return False
        return cb

    for keycode, name in CAMERA_HELD_KEYS.items():
        vis.register_key_action_callback(keycode, make_hold_callback(rig, name))
    for keycode, name in ARM_HELD_KEYS.items():
        vis.register_key_action_callback(keycode, make_hold_callback(arm, name))

    # ---- Tap (press-only) callbacks ------------------------------------
    def toggle_continuous(_vis):
        state["continuous"] = not state["continuous"]
        print(f"[continuous streaming] {'ON' if state['continuous'] else 'OFF'}")
        return False

    def capture_once(_vis):
        state["capture_once"] = True
        return False

    def gripper_open(_vis):
        state["gripper"] = min(255.0, state["gripper"] + 40.0)
        return False

    def gripper_close(_vis):
        state["gripper"] = max(0.0, state["gripper"] - 40.0)
        return False

    vis.register_key_callback(glfw.KEY_C, toggle_continuous)
    vis.register_key_callback(glfw.KEY_SPACE, capture_once)
    vis.register_key_callback(glfw.KEY_N, gripper_open)
    vis.register_key_callback(glfw.KEY_M, gripper_close)

    # ---- Per-frame animation callback: physics + camera + point cloud --
    def animation_callback(vis):
        t_now = time.time()
        dt = t_now - timing["t_prev"]
        timing["t_prev"] = t_now

        data.ctrl[7] = state["gripper"]  # actuator8: tendon-coupled gripper, 0-255

        rig.step(dt)
        arm.step(dt)
        mujoco.mj_step(model, data)

        do_capture = state["continuous"] or state["capture_once"]
        if do_capture and (t_now - timing["last_capture"]) >= min_period:
            points, colors = capture_pointcloud(
                model, data, renderer, "scene_cam", depth_trunc=args.depth_trunc)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            timing["last_capture"] = t_now
            state["capture_once"] = False
            if not view_initialized["done"] and len(points) > 0:
                vis.reset_view_point(True)
                view_initialized["done"] = True
        return False

    vis.register_animation_callback(animation_callback)

    print("Open3D window ready. Click it for keyboard focus.")
    print("  Camera: WASD/RF move, arrows yaw/pitch, Q/E roll")
    print("  Arm:    I/K +-X, J/L +-Y, U/O up/down (differential IK)")
    print("  Gripper: N open, M close")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
