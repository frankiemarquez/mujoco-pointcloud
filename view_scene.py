"""
view_scene.py

Opens MuJoCo's own native interactive viewer for this scene -- separate
from the point-cloud app. Use this just to look around the simulation with
the mouse and sanity-check the model (objects, arm, cameras).

This uses `mujoco.viewer.launch()` (the blocking, full-control viewer),
NOT `launch_passive()`. That distinction matters on macOS: `launch_passive`
hands control back to your own script loop while the GUI runs on the main
thread, which requires the special `mjpython` launcher. `launch()` instead
takes over the whole main thread itself and blocks until you close the
window -- so it works with plain `python3` on every platform, no mjpython
needed.

Run:
    python3 view_scene.py

Mouse controls (MuJoCo's defaults):
    Left-drag           orbit / rotate the camera
    Right-drag          pan
    Scroll wheel         zoom
    Ctrl + right-drag    move the camera target point
    Double-click a body  select it (shows contact forces etc. via the UI)

Keyboard (MuJoCo's built-in bindings, unrelated to our point-cloud app):
    Space                pause / resume simulation
    Backspace             reset simulation (back to the arm's default zero
                           pose, not the bent 'home' pose this script starts
                           at -- the nicer pose is set once at launch here,
                           not stored as an in-model keyframe; see
                           scene_setup.py for why)
    Tab                   toggle the left/right UI panels
"""
import os
import mujoco
import mujoco.viewer
from scene_setup import reset_to_home

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    reset_to_home(model, data)  # Panda in 'home' pose, objects settled onto table
    mujoco.viewer.launch(model, data)
