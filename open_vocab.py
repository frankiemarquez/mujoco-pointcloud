"""
open_vocab.py

Scores candidate 3D point-cloud segments (from `segmentation_utils.Segment`)
against an arbitrary open-vocabulary text query -- e.g. "green cube",
"the tall purple thing", "orange ball" -- and returns them ranked by how
well each one matches.

Two matcher backends:

  CLIPMatcher (real open-vocabulary understanding)
  --------------------------------------------------
  Uses CLIP (via `open_clip_torch`) to embed the text query and an image
  crop of each candidate segment, then ranks by cosine similarity. This is
  genuinely open-vocabulary: it understands arbitrary language, not just a
  fixed list of color/shape words.

  This needs `pip install torch open_clip_torch pillow` and downloads
  ~350MB of pretrained weights on first use. Both were too heavy to
  exercise inside the sandbox this project was built in (its network is
  restricted to package indexes, not model-weight hosts, and a full
  torch install pulls multiple GB of CUDA dependencies) -- so this class is
  implemented against the real `open_clip` API and is ready to run on any
  machine with normal internet access, but it was not runtime-tested
  during development. If you hit an API mismatch after a library update,
  the fix is almost always a one-line change in `CLIPMatcher.__init__`.

  HeuristicMatcher (dependency-free fallback)
  --------------------------------------------
  No ML, no extra installs -- works with what this project already
  depends on (numpy + open3d). Parses a color word and, more weakly, a
  shape word out of the query, and scores each segment by:
    - how close its mean point color is to the named color (primary signal,
      reliable), and
    - a rough curvature-based "roundness" estimate compared against the
      named shape (secondary signal, weak -- flat boxes vs. curved
      spheres/cylinders separate somewhat, but this is not a real shape
      classifier).
  This covers the common case ("red box", "the green sphere", "blue
  cylinder") without needing CLIP at all, and is what `open_vocab_query.py`
  falls back to automatically if CLIP/torch aren't installed.

`get_matcher()` picks CLIP if available, otherwise the heuristic -- callers
don't need to know which one they got.
"""
from __future__ import annotations
import re
import colorsys
import numpy as np
import open3d as o3d

try:
    import torch
    import open_clip
    from PIL import Image
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

    def _no_grad_noop(fn):
        """Stand-in for @torch.no_grad() so the module still imports (and
        CLIPMatcher's method body still parses) when torch isn't installed.
        CLIPMatcher.__init__ raises before this would ever actually run."""
        return fn


# ---------------------------------------------------------------------------
# CLIP-based matcher (real open-vocabulary matching)
# ---------------------------------------------------------------------------
class CLIPMatcher:
    """Zero-shot text<->image similarity via CLIP. Requires
    `pip install torch open_clip_torch pillow`; downloads pretrained
    weights on first use (needs internet)."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        if not HAS_CLIP:
            raise RuntimeError(
                "CLIPMatcher needs `torch`, `open_clip_torch`, and `pillow`. "
                "Install with: pip install torch open_clip_torch pillow")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @(torch.no_grad() if HAS_CLIP else _no_grad_noop)
    def score(self, query: str, segments, rgb_image: np.ndarray) -> np.ndarray:
        """Return a (len(segments),) array of cosine similarities between
        `query` and each segment's cropped image region."""
        text_tokens = self.tokenizer([query]).to(self.device)
        text_feat = self.model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        crops = [seg.crop_rgb(rgb_image) for seg in segments]
        images = torch.stack([
            self.preprocess(Image.fromarray(crop)) for crop in crops
        ]).to(self.device)
        image_feat = self.model.encode_image(images)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        sims = (image_feat @ text_feat.T).squeeze(-1)
        return sims.cpu().numpy()


# ---------------------------------------------------------------------------
# Dependency-free fallback matcher
# ---------------------------------------------------------------------------
_COLOR_WORDS = {
    "red":    np.array([0.85, 0.15, 0.15]),
    "green":  np.array([0.15, 0.75, 0.25]),
    "blue":   np.array([0.15, 0.35, 0.90]),
    "yellow": np.array([0.90, 0.80, 0.10]),
    "purple": np.array([0.60, 0.20, 0.75]),
    "violet": np.array([0.60, 0.20, 0.75]),
    "orange": np.array([0.90, 0.50, 0.10]),
    "white":  np.array([0.90, 0.90, 0.90]),
    "black":  np.array([0.10, 0.10, 0.10]),
    "gray":   np.array([0.5, 0.5, 0.5]),
    "grey":   np.array([0.5, 0.5, 0.5]),
}

# Empirically-calibrated curvature ranges (see dev notes / README) for this
# project's own primitive shapes -- a weak secondary signal, not a real
# shape classifier. "box"/"cube" ~ flattest, "sphere"/"ball" ~ most curved,
# cylinder/capsule in between.
_SHAPE_WORDS = {
    "box": "box", "cube": "box", "block": "box", "square": "box",
    "sphere": "sphere", "ball": "sphere", "round": "sphere",
    "cylinder": "cylinder", "tube": "cylinder", "can": "cylinder",
    "capsule": "capsule", "pill": "capsule",
}
_SHAPE_CURVATURE_TARGET = {"box": 0.08, "cylinder": 0.10, "capsule": 0.11, "sphere": 0.12}


def _hue_sat_similarity(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """Compare two RGB colors by hue + saturation (ignoring brightness/value).

    Plain RGB cosine similarity conflates e.g. "orange" and "brown" -- they
    point in nearly the same RGB direction, differing mainly in how dark/
    desaturated they are. Comparing in HSV and ignoring V (brightness)
    fixes that: orange (saturated) and brown (desaturated, same hue) come
    out clearly different once saturation is taken into account.
    """
    h_a, s_a, _ = colorsys.rgb_to_hsv(*np.clip(rgb_a, 0, 1))
    h_b, s_b, _ = colorsys.rgb_to_hsv(*np.clip(rgb_b, 0, 1))
    hue_dist = min(abs(h_a - h_b), 1.0 - abs(h_a - h_b))  # hue is circular
    hue_sim = 1.0 - 2.0 * hue_dist
    sat_sim = 1.0 - abs(s_a - s_b)
    return 0.7 * hue_sim + 0.3 * sat_sim


def _parse_query(query: str):
    words = re.findall(r"[a-zA-Z]+", query.lower())
    color = next((w for w in words if w in _COLOR_WORDS), None)
    shape = next((w for w in words if w in _SHAPE_WORDS), None)
    return color, (_SHAPE_WORDS[shape] if shape else None)


def _curvature_score(points: np.ndarray, k: int = 10) -> float:
    """Mean angular difference between each point's estimated normal and
    its k nearest neighbors' normals -- a rough local-curvature proxy.
    Flat surfaces (box faces) -> low; curved surfaces (spheres) -> higher."""
    if len(points) < k + 1:
        return 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals)
    tree = o3d.geometry.KDTreeFlann(pcd)

    # subsample for speed on large segments -- curvature is a smooth
    # statistic, doesn't need every single point.
    n = len(points)
    sample_idx = np.random.choice(n, size=min(n, 300), replace=False)

    diffs = []
    for i in sample_idx:
        _, idx, _ = tree.search_knn_vector_3d(points[i], k)
        ni = normals[i]
        for j in idx[1:]:
            cos = np.clip(abs(np.dot(ni, normals[j])), -1.0, 1.0)
            diffs.append(np.arccos(cos))
    return float(np.mean(diffs)) if diffs else 0.0


class HeuristicMatcher:
    """Dependency-free fallback: color (primary) + rough shape (secondary)
    keyword matching. No ML, no downloads -- runs anywhere numpy/open3d do."""

    def __init__(self, color_weight: float = 0.75, shape_weight: float = 0.25):
        self.color_weight = color_weight
        self.shape_weight = shape_weight

    def score(self, query: str, segments, rgb_image: np.ndarray = None) -> np.ndarray:
        color_name, shape_name = _parse_query(query)
        scores = np.zeros(len(segments))

        for i, seg in enumerate(segments):
            s = 0.0
            total_weight = 0.0

            if color_name is not None:
                target = _COLOR_WORDS[color_name]
                mean_color = seg.colors.mean(axis=0)
                # HSV hue+saturation comparison, NOT raw RGB cosine --
                # cosine similarity conflates orange/brown (same RGB
                # direction, different saturation); see _hue_sat_similarity.
                color_sim = _hue_sat_similarity(target, mean_color)
                s += self.color_weight * color_sim
                total_weight += self.color_weight

            if shape_name is not None:
                curvature = _curvature_score(seg.points)
                target_curv = _SHAPE_CURVATURE_TARGET[shape_name]
                # closer curvature to the target shape's typical value -> higher score
                shape_sim = max(0.0, 1.0 - abs(curvature - target_curv) / 0.05)
                s += self.shape_weight * shape_sim
                total_weight += self.shape_weight

            if total_weight == 0:
                # no recognized color/shape words at all -- can't say anything
                scores[i] = 0.0
            else:
                scores[i] = s / total_weight

        return scores


def get_matcher(prefer_clip: bool = True):
    """Return the best available matcher: CLIP if installed and usable,
    otherwise the dependency-free heuristic fallback. Returns
    (matcher, backend_name)."""
    if prefer_clip and HAS_CLIP:
        try:
            return CLIPMatcher(), "clip"
        except Exception as e:  # model download failed, no internet, etc.
            print(f"[open_vocab] CLIP unavailable ({e}); "
                  f"falling back to color/shape heuristic.")
    return HeuristicMatcher(), "heuristic"
