"""
segmentation_utils.py

Turns a raw colorized point cloud into a set of candidate OBJECT SEGMENTS,
each with:
  - its own (points, colors) subset
  - a 3D bounding box (axis-aligned and oriented)
  - the corresponding crop of the source RGB image (for scoring against an
    open-vocabulary text query in `open_vocab.py`)

Approach (classical, no learned segmentation model needed -- appropriate
for a synthetic tabletop scene with a handful of well-separated objects):
  1. Remove big supporting planes (floor, table top) via RANSAC, applied
     iteratively since there can be more than one dominant plane.
  2. Cluster what's left with DBSCAN -- each remaining blob is a candidate
     object instance.
  3. For each cluster, compute a bounding box and project the cluster's 3D
     points back to their source pixels (tracked all the way from
     `capture_pointcloud_with_pixels`) to get an image crop.

This is intentionally decoupled from *what* the objects are -- it just
proposes "here are the separate things in the scene"; matching one of them
to a text query like "green cube" is `open_vocab.py`'s job.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import open3d as o3d


@dataclass
class Segment:
    indices: np.ndarray          # indices into the original points/colors arrays
    points: np.ndarray           # (n, 3)
    colors: np.ndarray           # (n, 3) in [0, 1]
    pixels: np.ndarray           # (n, 2) (u, v) source pixel coords
    centroid: np.ndarray = field(init=False)
    aabb_min: np.ndarray = field(init=False)
    aabb_max: np.ndarray = field(init=False)
    image_bbox: tuple = field(init=False)  # (u_min, v_min, u_max, v_max)

    def __post_init__(self):
        self.centroid = self.points.mean(axis=0)
        self.aabb_min = self.points.min(axis=0)
        self.aabb_max = self.points.max(axis=0)
        u_min, v_min = self.pixels.min(axis=0)
        u_max, v_max = self.pixels.max(axis=0)
        self.image_bbox = (int(u_min), int(v_min), int(u_max), int(v_max))

    def crop_rgb(self, rgb: np.ndarray, pad: int = 6):
        """Return the RGB crop covering this segment, with a small margin
        so shape context around the object edge is preserved."""
        H, W = rgb.shape[:2]
        u0, v0, u1, v1 = self.image_bbox
        u0 = max(0, u0 - pad); v0 = max(0, v0 - pad)
        u1 = min(W - 1, u1 + pad); v1 = min(H - 1, v1 + pad)
        return rgb[v0:v1 + 1, u0:u1 + 1]

    def size(self):
        """Extent of the segment's bounding box (dx, dy, dz), in meters."""
        return self.aabb_max - self.aabb_min

    def open3d_bbox_lineset(self, color=(1.0, 0.0, 1.0)):
        """An Open3D LineSet drawing this segment's axis-aligned bounding box
        (a wireframe box), for visualization."""
        aabb = o3d.geometry.AxisAlignedBoundingBox(self.aabb_min, self.aabb_max)
        ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        ls.colors = o3d.utility.Vector3dVector([list(color)] * len(ls.lines))
        return ls

    def oriented_bbox(self):
        """Open3D OrientedBoundingBox, tighter than the AABB for rotated
        objects (e.g. a box/cylinder not aligned with world axes)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if len(self.points) < 4:
            return None
        try:
            return pcd.get_oriented_bounding_box()
        except RuntimeError:
            return None


def _remove_dominant_planes(points, colors, pixels, max_planes=2,
                             distance_threshold=0.012, ransac_n=3,
                             num_iterations=1000, min_plane_fraction=0.08):
    """Iteratively strip the largest planar surfaces (floor, table top).

    Stops early if the next-best plane covers less than
    `min_plane_fraction` of the remaining points, on the assumption that
    what's left is genuine objects, not another big flat surface.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    remaining_idx = np.arange(len(points))
    for _ in range(max_planes):
        if len(remaining_idx) < ransac_n * 5:
            break
        sub_pcd = pcd.select_by_index(remaining_idx)
        try:
            _, inlier_local = sub_pcd.segment_plane(
                distance_threshold, ransac_n, num_iterations)
        except RuntimeError:
            break
        if len(inlier_local) < min_plane_fraction * len(remaining_idx):
            break
        inlier_local = np.array(inlier_local)
        keep_mask = np.ones(len(remaining_idx), dtype=bool)
        keep_mask[inlier_local] = False
        remaining_idx = remaining_idx[keep_mask]

    return points[remaining_idx], colors[remaining_idx], pixels[remaining_idx]


def segment_objects(points: np.ndarray, colors: np.ndarray, pixels: np.ndarray,
                     max_planes: int = 2, plane_dist_thresh: float = 0.012,
                     cluster_eps: float = 0.03, cluster_min_points: int = 40,
                     min_extent: float = 0.015, max_extent: float = 0.35
                     ) -> list[Segment]:
    """Full pipeline: remove supporting planes, cluster the rest, package
    each cluster as a `Segment`.

    Args:
      points, colors, pixels: aligned (N,3), (N,3), (N,2) arrays as produced
                               by `capture_pointcloud_with_pixels`.
      max_planes: how many big planar surfaces to strip (floor + table = 2).
      plane_dist_thresh: RANSAC inlier distance (meters).
      cluster_eps: DBSCAN neighborhood radius (meters) -- objects closer
                   together than this will merge into one cluster.
      cluster_min_points: minimum cluster size to keep (filters noise).
      min_extent / max_extent: drop clusters whose largest bounding-box
                   dimension falls outside this range (meters). Plane
                   removal isn't perfect on a box-shaped table (it strips
                   flat faces but leaves things like the table's underside
                   or legs as separate blobs) -- this filters those out
                   without needing to know anything about what a "table"
                   is. Tune/relax for scenes with bigger target objects.

    Returns: list of `Segment`, one per detected candidate object.
    """
    obj_points, obj_colors, obj_pixels = _remove_dominant_planes(
        points, colors, pixels, max_planes=max_planes,
        distance_threshold=plane_dist_thresh)

    if len(obj_points) == 0:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_points)
    labels = np.array(pcd.cluster_dbscan(
        eps=cluster_eps, min_points=cluster_min_points, print_progress=False))

    segments = []
    for label in sorted(set(labels.tolist()) - {-1}):
        mask = labels == label
        seg = Segment(
            indices=np.nonzero(mask)[0],
            points=obj_points[mask],
            colors=obj_colors[mask],
            pixels=obj_pixels[mask],
        )
        largest_dim = seg.size().max()
        if not (min_extent <= largest_dim <= max_extent):
            continue
        segments.append(seg)
    return segments
