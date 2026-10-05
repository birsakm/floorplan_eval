"""Distribution distances between a generated set and a reference (GT) set."""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from .format import ROOM_TYPES


def _hist(values, bins):
    h = np.array([np.sum(values == b) for b in bins], dtype=float)
    return h / h.sum() if h.sum() > 0 else h


def stat_distances(gen_samples, gen_rooms, ref_samples, ref_rooms, metric_scale_known):
    """gen_/ref_samples: per-sample DataFrames; gen_/ref_rooms: per-room DataFrames."""
    out = {}
    bins = np.arange(0, max(gen_samples.n_rooms.max(), ref_samples.n_rooms.max()) + 1)
    out["dist_room_count_tv"] = 0.5 * np.abs(
        _hist(gen_samples.n_rooms.values, bins) - _hist(ref_samples.n_rooms.values, bins)).sum()
    gt = _hist(gen_rooms.type.values, ROOM_TYPES)
    rt = _hist(ref_rooms.type.values, ROOM_TYPES)
    out["dist_room_type_jsd"] = float(jensenshannon(gt, rt, base=2) ** 2) if gt.sum() and rt.sum() else np.nan
    out["dist_area_frac_w1"] = wasserstein_distance(gen_rooms.area_frac, ref_rooms.area_frac)
    out["dist_aspect_w1"] = wasserstein_distance(np.clip(gen_rooms.aspect, 1, 10), np.clip(ref_rooms.aspect, 1, 10))
    out["dist_convexity_w1"] = wasserstein_distance(gen_rooms.convexity, ref_rooms.convexity)
    if metric_scale_known:
        out["dist_room_area_m2_w1"] = wasserstein_distance(gen_rooms.area_m2, ref_rooms.area_m2)
        out["dist_total_area_m2_w1"] = wasserstein_distance(
            gen_samples.total_area_m2.dropna(), ref_samples.total_area_m2.dropna())
    else:
        out["dist_room_area_m2_w1"] = out["dist_total_area_m2_w1"] = np.nan
    return out


# ---------------------------------------------------------------- FID / KID

_INCEPTION = None


def _inception(device):
    global _INCEPTION
    if _INCEPTION is None:
        from pytorch_fid.inception import InceptionV3
        _INCEPTION = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    return _INCEPTION


def inception_features(images, device="cuda", batch=200):
    """images: iterable of PIL RGB images -> (N, 2048) float64 features."""
    import torch
    model = _inception(device)
    feats, buf = [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div(255).to(device)
        with torch.no_grad():
            feats.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        buf.clear()

    for img in images:
        buf.append(np.asarray(img.convert("RGB")))
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    return np.concatenate(feats).astype(np.float64)


def fid(f1, f2):
    from scipy import linalg
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kid(f1, f2, n_subsets=50, subset_size=500, seed=0):
    """Unbiased KID (polynomial kernel), mean and std over random subsets."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f1), len(f2))
    d = f1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))
