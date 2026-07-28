"""
demo_open_vocab_headless.py

Headless (no display needed) proof that open_vocab_query's full pipeline
works: runs several open-vocabulary queries against the scene and saves a
grid of images showing, for each query, the full point cloud with the
matched object highlighted and its 3D bounding box drawn -- as 2D
projections (matplotlib), so it renders without any GPU/display, mirroring
how `demo_flythrough.py` and `show_rgbd_pipeline.py` were verified earlier
in this project.

Run:
    MUJOCO_GL=egl python3 demo_open_vocab_headless.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from open_vocab_query import run_query

OUT_DIR = os.path.join(os.path.dirname(__file__), "open_vocab_proof")
os.makedirs(OUT_DIR, exist_ok=True)

QUERIES = ["red box", "green sphere", "blue cylinder", "purple capsule", "orange ball", "yellow cube"]


def draw_2d(ax, points, colors, matched_segments, title):
    dimmed = colors * 0.35 + 0.05
    ax.scatter(points[:, 0], points[:, 1], c=dimmed, s=0.5, zorder=1)
    for seg in matched_segments:
        ax.scatter(seg.points[:, 0], seg.points[:, 1], c=seg.colors, s=2.0, zorder=2)
        x0, y0 = seg.aabb_min[0], seg.aabb_min[1]
        x1, y1 = seg.aabb_max[0], seg.aabb_max[1]
        box = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                       fill=False, edgecolor="magenta", linewidth=2, zorder=3)
        ax.add_patch(box)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.axis("equal")


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, query in enumerate(QUERIES):
    print(f"\n{'='*60}\nQUERY: {query}\n{'='*60}")
    points, colors, matched, rgb = run_query(query, topk=1)
    if matched is None:
        continue
    draw_2d(axes[i], points, colors, matched,
            f'"{query}" -> matched segment '
            f'(color={np.round(matched[0].colors.mean(axis=0), 2)})')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "open_vocab_grid.png")
plt.savefig(out_path, dpi=130)
print(f"\nWrote {out_path}")
