"""
pointcloud_utils.py

Utilities to turn a MuJoCo RGB + depth camera pair into a colorized 3D point
cloud (in world coordinates), usable with Open3D.

MuJoCo camera convention (per mujoco.Renderer):
  - The camera's local frame has +X right, +Y up, -Z forward (OpenGL style).
  - `data.cam_xpos[id]`  -> camera position in world frame.
  - `data.cam_xmat[id]`  -> 3x3 rotation, camera-frame axes expressed in world
                            frame (columns are camera x/y/z axes in world).
  - Depth returned by `Renderer.render()` (after `enable_depth_rendering()`)
    is *metric* straight-line distance along the camera's viewing axis
    (i.e. already the perspective Z distance in meters, not a raw buffer
    value), with shape (H, W), dtype float32. Invalid/background pixels are
    typically at the renderer's zfar.
"""

from __future__ import annotations
import numpy as np
import mujoco


def camera_intrinsics(model: mujoco.MjModel, cam_id: int, width: int, height: int):
    """Compute a pinhole intrinsics matrix for a MuJoCo camera.

    MuJoCo cameras are specified with a vertical field of view (`fovy`, in
    degrees). Pixels are assumed square, so fx == fy in pixel units.
    """
    fovy = model.cam_fovy[cam_id]
    fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


def camera_pose(data: mujoco.MjData, cam_id: int):
    """Return (R_world_cam, t_world_cam): the camera's pose in world coords.

    R_world_cam columns are the camera's x/y/z axes expressed in the world
    frame; t_world_cam is the camera position in the world frame.
    """
    R = np.array(data.cam_xmat[cam_id]).reshape(3, 3)
    t = np.array(data.cam_xpos[cam_id])
    return R, t


def depth_rgb_to_pointcloud(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray,
                             R_world_cam: np.ndarray, t_world_cam: np.ndarray,
                             depth_trunc: float = 5.0, min_depth: float = 1e-3,
                             return_pixels: bool = False):
    """Convert an aligned depth + rgb image pair into a colorized point cloud.

    Args:
      depth: (H, W) float32 metric depth (distance along camera viewing axis).
      rgb:   (H, W, 3) uint8 color image, pixel-aligned with `depth`.
      K:     3x3 pinhole intrinsics (fx, fy, cx, cy) as produced by
             `camera_intrinsics`.
      R_world_cam, t_world_cam: camera pose in world frame (see `camera_pose`).
      depth_trunc: discard points farther than this (meters) -- removes the
                   background / "sky" pixels that render at zfar.
      min_depth: discard points closer than this (meters).
      return_pixels: if True, also return the (u, v) source pixel coordinate
                     for every point (needed to later crop the RGB image for
                     a subset of points, e.g. for open-vocabulary scoring).

    Returns:
      points: (N, 3) float64 array of world-frame XYZ points.
      colors: (N, 3) float64 array of RGB colors in [0, 1].
      pixels: (N, 2) int array of (u, v) pixel coords -- only if
              `return_pixels=True`.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    z = depth.astype(np.float64)
    valid = (z > min_depth) & (z < depth_trunc)

    # Pinhole unprojection in camera-local frame. MuJoCo/OpenGL camera frame:
    # +X right, +Y up, -Z forward -> forward distance z_forward = z,
    # so the camera-frame Z coordinate is -z_forward.
    x_cam = (uu - cx) * z / fx
    y_cam = -(vv - cy) * z / fy   # image v grows downward, camera Y is up
    z_cam = -z                    # camera looks down -Z

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    pts_cam = pts_cam[valid]
    cols = (rgb.astype(np.float64) / 255.0)[valid]

    # Transform camera-frame points into world frame:
    # p_world = R_world_cam @ p_cam + t_world_cam
    pts_world = pts_cam @ R_world_cam.T + t_world_cam

    if return_pixels:
        pix = np.stack([uu, vv], axis=-1)[valid]  # (N, 2) as (u, v)
        return pts_world, cols, pix
    return pts_world, cols


def render_rgbd(renderer: "mujoco.Renderer", data: mujoco.MjData, cam_name: str):
    """Render an aligned (rgb, depth) pair for the given camera name."""
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    return rgb, depth


def capture_pointcloud(model: mujoco.MjModel, data: mujoco.MjData,
                        renderer: "mujoco.Renderer", cam_name: str,
                        depth_trunc: float = 5.0):
    """One-shot: render RGB+D from `cam_name` and return a world-frame
    colorized point cloud as (points, colors) numpy arrays."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    rgb, depth = render_rgbd(renderer, data, cam_name)
    K = camera_intrinsics(model, cam_id, renderer.width, renderer.height)
    R, t = camera_pose(data, cam_id)
    return depth_rgb_to_pointcloud(depth, rgb, K, R, t, depth_trunc=depth_trunc)


def capture_pointcloud_with_pixels(model: mujoco.MjModel, data: mujoco.MjData,
                                    renderer: "mujoco.Renderer", cam_name: str,
                                    depth_trunc: float = 5.0):
    """Like `capture_pointcloud`, but also returns the source (u, v) pixel
    for every 3D point and the full RGB image itself. Needed for
    open-vocabulary segmentation, where a 3D cluster of points must be
    traced back to a 2D image crop to be scored against a text query.

    Returns:
      points: (N, 3) world-frame XYZ.
      colors: (N, 3) RGB in [0, 1].
      pixels: (N, 2) int (u, v) source pixel coordinates, aligned with points.
      rgb:    (H, W, 3) uint8 the full rendered RGB image (for cropping).
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    rgb, depth = render_rgbd(renderer, data, cam_name)
    K = camera_intrinsics(model, cam_id, renderer.width, renderer.height)
    R, t = camera_pose(data, cam_id)
    points, colors, pixels = depth_rgb_to_pointcloud(
        depth, rgb, K, R, t, depth_trunc=depth_trunc, return_pixels=True)
    return points, colors, pixels, rgb
