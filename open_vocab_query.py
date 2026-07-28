"""
open_vocab_query.py

Given an open-vocabulary text query (e.g. "green cube", "the orange ball"),
this finds the best-matching object in the current RGB-D observation and
displays it: the full colorized point cloud (dimmed), the matched segment
highlighted in its true color, and a 3D bounding box drawn around it.

Pipeline:  RGB-D capture -> colorized point cloud -> segmentation into
candidate objects -> score each candidate against the text query -> show
the winner.

Run:
    python3 open_vocab_query.py "green sphere"
    python3 open_vocab_query.py "red box" --topk 3      # show top 3 matches
    python3 open_vocab_query.py "blue cylinder" --camera end_effector_camera

Uses CLIP for real open-vocabulary matching if `torch` + `open_clip_torch`
are installed (`pip install torch open_clip_torch pillow`); otherwise
automatically falls back to a dependency-free color/shape heuristic (see
`open_vocab.py` for details on both).
"""
import os
import argparse

import numpy as np
import mujoco
import open3d as o3d

from pointcloud_utils import capture_pointcloud_with_pixels
from segmentation_utils import segment_objects
from open_vocab import get_matcher
from scene_setup import reset_to_home, disable_shadows

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")


def run_query(query: str, camera: str = "scene_cam", topk: int = 1,
              width: int = 640, height: int = 480, settle_steps: int = 300,
              depth_trunc: float = 4.0):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    reset_to_home(model, data, settle_steps=settle_steps)  # Panda home pose, objects settled
    disable_shadows(model)  # ~5x faster offscreen capture (see scene_setup.py)

    renderer = mujoco.Renderer(model, height=height, width=width)
    points, colors, pixels, rgb = capture_pointcloud_with_pixels(
        model, data, renderer, camera, depth_trunc=depth_trunc)

    segments = segment_objects(points, colors, pixels)
    if not segments:
        print("No candidate objects found (check camera view / depth_trunc).")
        return None, None, None, None

    matcher, backend = get_matcher()
    print(f"[open_vocab_query] matcher backend: {backend}")

    scores = matcher.score(query, segments, rgb)
    order = np.argsort(scores)[::-1]

    print(f"\nQuery: \"{query}\"  ({len(segments)} candidate objects found)")
    print("-" * 60)
    for rank, idx in enumerate(order[:max(topk, 5)]):
        seg = segments[idx]
        marker = "  <-- MATCH" if rank < topk else ""
        print(f"  #{rank+1}  segment {idx:2d}  score={scores[idx]:.3f}  "
              f"mean_color={np.round(seg.colors.mean(axis=0), 2)}  "
              f"size={np.round(seg.size(), 3)}{marker}")

    best_indices = order[:topk]
    return points, colors, [segments[i] for i in best_indices], rgb


def visualize(points, colors, matched_segments, window_name="Open-Vocabulary Query Result"):
    """Full cloud (dimmed) + matched segment(s) highlighted + 3D bbox."""
    geometries = []

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(points)
    # dim the background cloud so the match visually pops
    dimmed = colors * 0.35 + 0.05
    full_pcd.colors = o3d.utility.Vector3dVector(dimmed)
    geometries.append(full_pcd)

    for seg in matched_segments:
        match_pcd = o3d.geometry.PointCloud()
        match_pcd.points = o3d.utility.Vector3dVector(seg.points)
        match_pcd.colors = o3d.utility.Vector3dVector(seg.colors)  # true color
        geometries.append(match_pcd)

        obb = seg.oriented_bbox()
        if obb is not None:
            obb.color = (1.0, 0.0, 1.0)  # magenta
            geometries.append(obb)
        geometries.append(seg.open3d_bbox_lineset(color=(1.0, 1.0, 0.0)))

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
    o3d.visualization.draw_geometries(geometries, window_name=window_name,
                                       width=1000, height=750)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help='e.g. "green cube", "the red box"')
    parser.add_argument("--camera", default="scene_cam", choices=["scene_cam", "end_effector_camera"])
    parser.add_argument("--topk", type=int, default=1,
                         help="how many top-scoring segments to highlight")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    points, colors, matched, rgb = run_query(
        args.query, camera=args.camera, topk=args.topk,
        width=args.width, height=args.height)

    if matched:
        visualize(points, colors, matched)
