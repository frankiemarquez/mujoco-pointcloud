# MuJoCo Colorized Point Cloud Playground

A [MuJoCo](https://github.com/google-deepmind/mujoco) scene with a few
objects, a real **Franka Emika Panda** arm (from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
extended to match the `attachment_site`/`end_effector_camera` pattern from
Kevin Zakka's [mjctrl](https://github.com/kevinzakka/mjctrl) manipulation
tutorial), and an RGB-D camera pipeline that produces a **colorized point
cloud**, visualized live in [Open3D](https://www.open3d.org/) and steerable
from the keyboard — plus open-vocabulary object segmentation on top.

The ground/skybox styling and the differential-IK demo (`panda_ik_demo.py`)
follow the coding patterns from the MuJoCo basics tutorial and the
manipulation tutorial this project is based on (checker-texture floor
materials, `mj_jacSite` + pseudo-inverse IK, `mjv`-scene trace capsules,
etc.) — see "Replicating the tutorials' style" below for specifics on what
was carried over verbatim vs. adapted.

## Contents

| File | Purpose |
|---|---|
| `scene.xml` | The MuJoCo model: tutorial-style checker floor/skybox, table, 6 free objects, the Panda arm (via `<include>`), an `ik_target` mocap body, and a keyboard-controllable "point-cloud camera" mounted on its own mocap body. |
| `franka_emika_panda/` | The Panda model itself (from mujoco_menagerie), with `attachment_site` and `end_effector_camera` added to its `hand` body. See "Notes on the Panda integration" for two real bugs found/fixed while wiring this in. |
| `scene_setup.py` | `reset_to_home()` — the *correct* way to reset this scene (Panda in its bent 'home' pose, objects at their real starting positions). Don't use `mj_resetDataKeyframe` on this model — see the module docstring. |
| `pointcloud_utils.py` | Core RGB-D → colorized point cloud math (intrinsics, camera pose, unprojection). |
| `camera_rig.py` | A free-flying camera controller (`CameraRig`) that turns held keys into a live pose written to the mocap body each frame. |
| `arm_control.py` | A keyboard-driven Cartesian arm controller (`ArmIKController`) -- jogs a target position, tracks it via damped-least-squares differential IK each frame. |
| `interactive_pointcloud.py` | **Main app.** One Open3D window: live keyboard-controlled point cloud + gripper control. |
| `view_scene.py` | MuJoCo's own native viewer (mouse-controlled) for just looking around the model, including a Control panel to jog the Panda's joints directly. |
| `panda_ik_demo.py` | Differential IK demo in the style of the manipulation tutorial: a circular target trajectory tracked via `mj_jacSite` + pseudo-inverse, with target/end-effector traces drawn in the scene. Writes `panda_ik_demo.mp4`. |
| `test_capture.py` | Minimal headless smoke test: one RGB-D capture → PNG + `.ply`. |
| `demo_flythrough.py` | Headless (no display needed) demo: scripts a sequence of "keyboard" moves, captures a point cloud at each pose, merges them, and saves PNG sanity-check plots — this is how the pipeline was verified in this sandbox. |
| `show_rgbd_pipeline.py` | Visual proof of RGB+depth->point-cloud extraction: saves the RGB image, colorized depth image, and point cloud side by side. |
| `test_pointcloud_extraction.py` | Correctness test for the extraction interface itself: checks output shapes/dtypes, and verifies extracted colors actually match known object colors (ground truth taken from the simulator). |
| `segmentation_utils.py` | Segments the point cloud into candidate objects: removes supporting planes (floor/table) via RANSAC, clusters what's left via DBSCAN, and packages each cluster as a `Segment` with a 3D bounding box + a traceable RGB crop. |
| `open_vocab.py` | Scores segments against an open-vocabulary text query. Real CLIP-based matching if `torch`+`open_clip_torch` are installed; a dependency-free color/shape heuristic fallback otherwise. |
| `open_vocab_query.py` | **Main open-vocabulary demo.** `python3 open_vocab_query.py "green cube"` -> finds the best-matching object and shows it in Open3D: full cloud dimmed, match highlighted in true color, 3D bounding box drawn around it. |
| `demo_open_vocab_headless.py` | Headless proof that the open-vocab pipeline works: runs several queries and saves a grid of 2D projections showing each match + bounding box. |
| `demo_output/` | Output of `demo_flythrough.py`: per-frame RGB + `.ply` clouds, merged cloud, and projection plots. |
| `proof/` | Output of `show_rgbd_pipeline.py`. |
| `open_vocab_proof/` | Output of `demo_open_vocab_headless.py`. |

## Setup

```bash
pip install -r requirements.txt
```

## Running the interactive demo (requires a display)

```bash
python3 interactive_pointcloud.py
```

(On Linux without a working GLFW/X11/Wayland session, try
`MUJOCO_GL=egl python3 interactive_pointcloud.py` — though EGL is really for
*offscreen* rendering, so if you have no display at all the interactive app
can't open a window regardless of backend; use `demo_flythrough.py` instead
for a fully headless run.)

Everything happens in **one Open3D window**: the physics simulation, camera
control, and live point-cloud rendering all run inside Open3D's own event
loop (`register_animation_callback` / `register_key_action_callback`). There
is no separate MuJoCo GUI window.

> **Why not a MuJoCo viewer window too?** An earlier version of this project
> also opened a native `mujoco.viewer` window. On macOS, that viewer insists
> on owning the real OS main thread, which requires launching the whole
> script via MuJoCo's special `mjpython` wrapper — and Open3D's window
> *also* wants the real main thread, so the two collide in one process
> (crashes / `mjpython` requirement errors). Since the actual goal is
> "keyboard control + Open3D point-cloud view," this version drops the
> MuJoCo GUI window entirely. You still get the full simulation (physics,
> objects, arm) — you just don't get a second window showing MuJoCo's own
> render of it. Everything needed to inspect the scene is visible in the
> point cloud itself.

### Controls (click the Open3D window first, for keyboard focus)

| Key | Action |
|---|---|
| `W` / `S` | Move point-cloud camera forward / backward (held) |
| `A` / `D` | Strafe left / right (held) |
| `R` / `F` | Move up / down (held) |
| `←` / `→` | Yaw (held) |
| `↑` / `↓` | Pitch (held) |
| `Q` / `E` | Roll (held) |
| `I` / `K` | Move **arm** end effector +X / -X (held) |
| `J` / `L` | Move **arm** end effector -Y / +Y (held) |
| `U` / `O` | Move **arm** end effector up / down (held) |
| `N` / `M` | Open / close the gripper (Panda's tendon-coupled fingers, ctrl range 0-255) |
| `C` | Toggle continuous point-cloud streaming (on by default) |
| `Space` | Force one capture (useful if streaming is paused) |
| `Esc` | Quit |

`W/A/S/D/R/F/Q/E`, the arrow keys, and `I/J/K/L/U/O` are all **held-key**
controls (powered by Open3D's `register_key_action_callback`, which reports
press/repeat/release — not just taps), so movement is smooth and continuous
while a key is down.

**Arm control** (`arm_control.ArmIKController`) works by jogging a Cartesian
target position and running one differential-IK step every frame to drive
the Panda's 7 arm joints toward it — same core math as `panda_ik_demo.py`
and, further upstream, Kevin Zakka's
[mjctrl](https://github.com/kevinzakka/mjctrl) `diffik.py`: position error →
`mj_jacSite` → damped least-squares → `mj_integratePos`. Unlike
`panda_ik_demo.py`'s plain pseudo-inverse (fine for a gentle scripted
circle), this uses **damped** least-squares (Buss 2009 — the same paper
mjctrl cites) since keyboard jogging is more likely to push toward
workspace edges / near-singular configurations, where a plain
pseudo-inverse can blow up. The target position is clamped to a
comfortable workspace box (`ArmIKController.bounds`) so you can't drive it
into the floor or out of reach.

**Null-space control** (matching mjctrl's `diffik_nullspace.py`, not just
`diffik.py`) is also included: the Panda has 7 joints but a position
target only constrains 3, so infinitely many joint configurations reach
the same point — plain IK has no preference among them, so jogging the
target around can drift the arm into a valid-but-visually-awkward
"elbow-flipped" pose. `ArmIKController` adds a secondary task that pulls
joints back toward the natural home configuration, projected into
whatever directions don't interfere with tracking the target (with wrist
joints weighted more strongly, since those visibly rotate the gripper).
This is a real trade-off, not a free lunch: because the *damped*
pseudo-inverse's null-space projector is only approximate (not exact, the
way a plain Moore-Penrose pseudo-inverse's would be), a strong null-space
gain measurably leaks into tracking accuracy. The default gain here was
tuned to keep tracking error comparable to the no-null-space baseline
while still visibly improving posture — turn `nullspace_gain` up for a
more consistently "home-like" pose at the cost of some tracking precision,
or down for tighter tracking at the cost of occasional awkward poses.

The gripper keys moved from `J`/`K` in an earlier version of this project
to `N`/`M`, to free up `I/J/K/L` for the arm's now-classic four-direction
jog layout.

The point cloud updates at up to `--stream_hz` (default 12 Hz); use
`--width/--height` to trade resolution for speed, e.g.:

```bash
python3 interactive_pointcloud.py --width 320 --height 240 --stream_hz 15
```

## Headless verification (what was actually run while building this)

> **Platform note:** `MUJOCO_GL=egl` below is a Linux-only offscreen
> rendering backend. On macOS/Windows, just drop that prefix and run the
> script plain (e.g. `python3 test_capture.py`) — the scripts only apply
> the `egl` default when `sys.platform` is Linux, so they're safe to run
> as-is on any OS.

This was developed in a container with **no display** attached, so the
interactive dual-window app (`interactive_pointcloud.py`) could not be
launched end-to-end here — MuJoCo's viewer and Open3D's window both need a
real windowing system (GLFW/X11/Wayland). Everything else was verified
directly:

```bash
MUJOCO_GL=egl python3 test_capture.py       # one-shot RGB/D -> point cloud
MUJOCO_GL=egl python3 demo_flythrough.py    # scripted multi-pose fly-through
```

`demo_flythrough.py` exercises the **exact same code path** the interactive
app uses (`CameraRig.step()`, `capture_pointcloud()`), just driving the
"keyboard" programmatically instead of from real key events, and confirmed:
- RGB and depth render correctly via MuJoCo's offscreen EGL renderer.
- The unprojected point clouds are correctly colored and spatially
  consistent across different camera poses (see
  `demo_output/merged_cloud_projections.png`).
- The camera rig's forward/strafe/yaw/pitch math moves the camera as
  expected.

On a machine with a GPU + display, just run `interactive_pointcloud.py`
directly — no code changes needed.

## How the point cloud math works

MuJoCo's `Renderer` gives you, per camera:
- an RGB image (`H x W x 3`, uint8), and
- (after `enable_depth_rendering()`) a **metric depth image** (`H x W`,
  float32) — the true perspective distance along the camera's viewing axis,
  not a normalized depth-buffer value.

`pointcloud_utils.py` then:
1. Builds a pinhole intrinsics matrix from the camera's vertical FOV
   (`model.cam_fovy`) and the image size (MuJoCo assumes square pixels).
2. Unprojects every pixel into the camera's local frame, respecting MuJoCo's
   OpenGL-style camera convention (`+X` right, `+Y` up, `-Z` forward).
3. Transforms into world coordinates using `data.cam_xpos` / `data.cam_xmat`
   (available for any camera, including one on a moving mocap body).
4. Filters out background pixels beyond `depth_trunc` meters.

The result is an `(N, 3)` world-frame XYZ array plus an aligned `(N, 3)`
RGB-in-`[0,1]` color array — exactly what `open3d.geometry.PointCloud` wants.

## The scene

- **Objects:** a red box, green sphere, blue cylinder, yellow box, purple
  capsule, and orange sphere — all free bodies on a small table, so they
  settle under gravity and can be nudged/grasped by the arm.
- **Robot:** a real **Franka Emika Panda** (7-DOF arm + 2-finger gripper),
  pulled from mujoco_menagerie via `<include file="franka_emika_panda/panda.xml"/>`.
  Extended (in this project's own copy of that file, not the stock one)
  with an `attachment_site` and `end_effector_camera` on its `hand` body,
  matching the pattern used in Kevin Zakka's mjctrl tutorial scenes. Driven
  by 8 actuators: `actuator1`-`actuator7` (arm joints, position-like PD via
  `biastype="affine"`) and `actuator8` (both fingers, tendon-coupled,
  ctrl range 0-255: 0=closed, 255=open).
- **`ik_target`:** a mocap body for the differential-IK demo
  (`panda_ik_demo.py`), same pattern as mjctrl's own scene files. Not
  driven by anything in the interactive app — only `panda_ik_demo.py`
  moves it.
- **Point-cloud camera (`scene_cam`):** mounted on a `mocap` body (`cam_rig`)
  so its pose can be set directly, frame-by-frame, from keyboard input,
  completely decoupled from physics.

## Notes on the Panda integration (two real bugs, and how they were fixed)

Wiring a full standalone mujoco_menagerie robot into a larger scene via
`<include>` hit two genuine MuJoCo gotchas worth knowing about if you do
this yourself:

1. **Mesh paths double-prefix.** When you `<include>` a full
   `<mujoco>`-rooted file (not just a body fragment) from a subdirectory,
   MuJoCo prefixes that file's own mesh `file=` references with its
   subdirectory *relative to the including file*, and then *also* prepends
   the included file's own `meshdir` compiler attribute in front of that —
   giving a broken doubled path (e.g. `assets/franka_emika_panda/link0.stl`
   instead of `franka_emika_panda/assets/link0.stl`). Fixed here by
   flattening `franka_emika_panda/assets/*` up into `franka_emika_panda/`
   directly and dropping the `meshdir` attribute entirely, so MuJoCo's
   default (same directory as the XML file) resolves correctly.

2. **Keyframes silently corrupt free-object placement.** The stock Panda
   file ships a `home` keyframe with a 9-value `qpos` (7 arm joints + 2
   fingers) — correct when the Panda is loaded alone. Once merged into this
   larger scene (51 qpos total: 9 for the Panda + 7 per free object × 6
   objects), calling `mj_resetDataKeyframe` pads the missing 42 values with
   a generic zero pose instead of each object's real declared position,
   teleporting every object to the origin — which then explodes into a
   scattered mess once physics resolves the resulting interpenetration.
   Fixed by deleting the keyframe from this project's copy of `panda.xml`
   and instead reproducing the same "home" pose in code, in
   `scene_setup.reset_to_home()`, which only ever touches the Panda's own
   9 qpos entries and leaves objects at their correct compiler-generated
   defaults (`qpos0`). **Always use `reset_to_home()` to initialize this
   scene** — never call `mj_resetDataKeyframe` on it directly.

## Replicating the tutorials' style

- **Ground/skybox:** `scene.xml`'s checker floor material (`grid` texture +
  material, gradient skybox) follows the exact same pattern as the basics
  tutorial's `tippe_top` / free-body examples.
- **`panda_ik_demo.py`** replicates the manipulation tutorial's differential
  IK example close to verbatim: the same `circle()` target trajectory
  function, the same `mju_negQuat`/`mju_mat2Quat`/`mju_mulQuat`/
  `mju_quat2Vel` orientation-error algebra, the same `mj_jacSite` →
  `np.linalg.pinv` → `mj_integratePos` IK step, and the same
  `add_visual_capsule`/`modify_scene` trace-drawing helpers. Two
  adaptations were required because this scene has more than one robot's
  worth of qpos: `data.ctrl = q` (tutorial) becomes `data.ctrl[:7] = q[:7]`
  (only the arm's own actuators — actuator8 is a separate gripper), and
  clipping to joint limits is scoped to `q[:7]` / `model.jnt_range[:7]`
  rather than the whole state. One MuJoCo API rename was also needed:
  `mjv_makeConnector` → `mjv_connector` in the installed mujoco version
  here, with a slightly different argument signature (arrays instead of 6
  separate floats).
- Both tutorials use `mediapy` (`media.show_image` / `media.show_video`) for
  output inside a notebook; `panda_ik_demo.py` uses the same library's
  `media.write_video` instead, since this project runs as plain scripts,
  not notebook cells. Note: the Panda's real mesh geometry is far heavier
  than the primitive-geom arm this project used before, so each rendered
  frame costs roughly ~0.9s here — `panda_ik_demo.py`'s duration/framerate
  are kept modest (5s @ 8fps) accordingly; turn them up freely on a
  machine with a GPU.

## Open-vocabulary object query

Given a free-text description like `"green cube"`, find the matching object
in the current view and display it with a 3D bounding box:

```bash
python3 open_vocab_query.py "green sphere"
python3 open_vocab_query.py "red box" --topk 3       # highlight top 3 matches
python3 open_vocab_query.py "blue cylinder" --camera end_effector_camera
```

This opens an Open3D window showing the full point cloud (dimmed), the
matched object highlighted in its true color, and both an axis-aligned
(yellow) and oriented (magenta) 3D bounding box around it. It also prints a
ranked score table to the terminal so you can see *why* it picked what it
picked:

```
Query: "green sphere"  (10 candidate objects found)
------------------------------------------------------------
  #1  segment  6  score=0.917  mean_color=[0.14 0.64 0.22]  size=[0.082 0.077 0.076]  <-- MATCH
  #2  segment  2  score=0.654  mean_color=[0.79 0.47 0.11]  size=[0.071 0.057 0.066]
  ...
```

**Pipeline:** RGB-D capture → colorized point cloud (existing interface) →
`segmentation_utils.py` removes the floor/table planes (RANSAC) and
clusters what's left (DBSCAN) into candidate objects → `open_vocab.py`
scores each candidate against your text query → the best match gets
highlighted and boxed.

**Two matcher backends** (picked automatically, no config needed):

- **CLIP** (`open_vocab.CLIPMatcher`) — genuine open-vocabulary matching:
  embeds your text and each candidate's image crop with CLIP and ranks by
  cosine similarity. Understands arbitrary language, not just a fixed word
  list. Needs `pip install torch open_clip_torch pillow` (downloads
  ~350MB of weights on first use, needs internet). **This path is
  implemented against the real `open_clip` API but was not runtime-tested
  in the sandbox this project was built in** — its network only allows
  package indexes, not model-weight hosts, and a full `torch` install
  pulls multiple GB of CUDA dependencies. It should work as-is on a normal
  machine with internet; if a library API changed since writing, the fix
  is almost always a one-line change in `CLIPMatcher.__init__`.
- **Heuristic fallback** (`open_vocab.HeuristicMatcher`) — no extra
  installs, works with what's already here (numpy + open3d). Parses a
  color word (primary signal — compares in HSV hue+saturation space, so it
  doesn't confuse e.g. "orange" with a brown table leg) and a shape word
  (secondary, weaker signal — a local-curvature estimate roughly separates
  flat boxes from curved spheres/cylinders, but this is *not* a real shape
  classifier). This is what runs automatically if CLIP/torch aren't
  installed, and is what was used to verify the whole pipeline in this
  sandbox — see `demo_open_vocab_headless.py` and `open_vocab_proof/`,
  where it correctly identified all 6 objects in the scene from color+shape
  queries alone.

**Headless verification:** `MUJOCO_GL=egl python3 demo_open_vocab_headless.py`
runs 6 queries ("red box", "green sphere", "blue cylinder", "purple
capsule", "orange ball", "yellow cube") end-to-end and saves a 2D-projection
proof grid — all 6 matched correctly with clear score margins over the
next-best candidate.

## Differential IK demo

```bash
MUJOCO_GL=egl python3 panda_ik_demo.py    # headless (writes panda_ik_demo.mp4)
python3 panda_ik_demo.py                   # also fine with a display
```

Drives the Panda's end effector around a small circle above the table using
Jacobian-based differential IK (same math as the manipulation tutorial —
see "Replicating the tutorials' style" above), drawing the target path
(blue) and actual end-effector path (red) as trace capsules. Saves an mp4
via `mediapy`. Typical tracking error is on the order of 1-2cm with this
simple pseudo-inverse approach (no null-space or OSC refinement, unlike the
tutorial's more advanced examples — those are straightforward to port over
using the same `jac`/`error` setup already here).

## Attribution

The Franka Emika Panda model (`franka_emika_panda/`) is from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
licensed Apache 2.0 (see `franka_emika_panda/LICENSE`), with
`attachment_site`/`end_effector_camera` added in this project's own copy —
see "Notes on the Panda integration" above. The `attachment_site` naming
pattern and `ik_target` mocap-body convention follow Kevin Zakka's
[mjctrl](https://github.com/kevinzakka/mjctrl).

## Extending it

- Swap `demo_flythrough.py`'s scripted key sequence for your own to test new
  behaviors headlessly.
- Point `capture_pointcloud(..., "end_effector_camera")` instead of
  `"scene_cam"` to get an eye-in-hand point cloud as the arm moves.
- `panda_ik_demo.py` already drives the arm to a target via differential
  IK — extend its `circle()` trajectory into a full pick-and-place by
  closing the gripper (`data.ctrl[7] = 0`) once near an object and opening
  it (`data.ctrl[7] = 255`) once above a drop location. All 6 objects are
  already sized to be graspable by the Panda's real gripper.
- `view_scene.py`'s Control panel (right-side UI, press `Tab` if hidden)
  lets you jog `actuator1`-`actuator8` directly with sliders — the fastest
  way to explore the arm's range of motion without writing any code.
