"""FID / KID between a generated set and a reference set, on Inception-V3 features of rendered layouts.

Render with fpeval.render.render_layout, the same way for both sets: the scores depend on the renderer.
"""
from multiprocessing import Pool

import numpy as np

_INCEPTION = {}
WALL_GAP_CLOSE_M = 0.3  # wall gaps up to twice this are filled before rendering (RPLAN-image sources: up to 0.6 m)


def default_device():
    """cuda if this torch build has kernels for the GPU (the fpe env's torch 2.4 has none for Blackwell), else cpu."""
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return "cuda"
    return "cpu"


def _inception(device):
    if device not in _INCEPTION:
        from pytorch_fid.inception import InceptionV3
        _INCEPTION[device] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    return _INCEPTION[device]


def inception_features(images, device=None, batch=100):
    """images: iterable of PIL RGB images -> (N, 2048) float64 features (pool3, as in pytorch-fid)."""
    import torch
    device = device or default_device()
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


def _render_file(args):
    from .format import load_sample
    from .render import render_layout
    path, scale = args
    return render_layout(load_sample(path), close=WALL_GAP_CLOSE_M / scale)


def layout_features(files, cache, scale=1.0, workers=16):
    """Inception features of render_layout(file) for files (scale: m per unit), cached in cache (.npz by path)."""
    have = dict(np.load(cache).items()) if cache.exists() else {}
    todo = [str(f) for f in files if str(f) not in have]
    if todo:
        with Pool(workers) as pool:
            imgs = pool.map(_render_file, [(f, scale) for f in todo], chunksize=8)
        have.update(zip(todo, inception_features(imgs)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **have)
    return np.stack([have[str(f)] for f in files])
