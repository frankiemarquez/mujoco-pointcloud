"""
scene_setup.py

Shared helper for loading `scene.xml` into a known-good initial state.

Why this exists (a real bug worth documenting, not just working around
silently): `franka_emika_panda/panda.xml` ships a `home` keyframe with a
9-value qpos vector (7 arm joints + 2 fingers) -- correct for the Panda
loaded on its own. Once merged (via `<include>`) into our larger scene,
the model's true `nq` is 51 (9 for the Panda + 7 per free object x 6
objects). Calling `mujoco.mj_resetDataKeyframe(model, data, 0)` on the
*merged* model pads the keyframe's missing 42 values with a generic zero
pose (position (0,0,0), quaternion (1,0,0,0)) instead of each object's
actual declared starting pose -- silently teleporting every free object to
the world origin, which then explodes into a scattered mess once physics
starts resolving the resulting interpenetration.

The fix: never call `mj_resetDataKeyframe` on this merged model. Instead,
start from `mj_forward` on the compiler's own defaults (`qpos0`), which
correctly places every object at its XML-declared position, and only then
manually overwrite the Panda's own 9 qpos entries with the desired "home"
arm configuration. Object qpos entries are never touched.
"""
import numpy as np
import mujoco

# The Panda's first 9 generalized coordinates: 7 arm joints + 2 fingers.
# Matches the `home` keyframe shipped in franka_emika_panda/panda.xml.
PANDA_HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
PANDA_HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])
PANDA_NQ = 9  # number of Panda generalized coordinates (must match above)


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData, settle_steps: int = 300):
    """Reset `data` to: Panda in its bent 'home' pose, every free object at
    its XML-declared starting position (NOT via mj_resetDataKeyframe -- see
    module docstring), then step physics so objects settle onto the table.
    """
    mujoco.mj_resetData(model, data)  # qpos = qpos0 (correct object placement)
    data.qpos[:PANDA_NQ] = PANDA_HOME_QPOS
    data.ctrl[:PANDA_NQ - 1] = PANDA_HOME_CTRL  # 8 actuators, not 9 (fingers share one)
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    return data
