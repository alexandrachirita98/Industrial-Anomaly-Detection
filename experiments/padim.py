"""PaDiM (Defard et al., 2020, arXiv:2011.08785) - patch distribution modelling.

One of the six unsupervised baselines benchmarked on AutoVI by Carvalho et al.
(2024), Computers in Industry 163, 104151. Chosen because "training" is a single
forward pass plus a mean and a covariance: no optimiser, no epochs, no GPU needed.

Trains on NORMAL images only. Use data/partitions/autovi_official (--mode official),
never the supervised re-split, or the AUROC is not comparable with the paper.

For every position in a 32x32 patch grid we fit one Gaussian over the normal
training images, then score a test image by the Mahalanobis distance of each of
its patches from the Gaussian at that same position. The maximum over the grid is
the image-level anomaly score; the grid itself is a free localisation heatmap.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # scipy arrives with scikit-learn, but do not hard-require it
    gaussian_filter = None

# ImageNet statistics: the frozen backbone was trained with these, so inputs
# must be normalised the same way. Skipping this is the usual cause of AUROC ~0.5.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Regularisation added to every covariance before inversion. The paper's value;
# without it np.linalg.inv raises LinAlgError on near-singular matrices.
COV_EPS = 0.01


def _blur(m: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of one (H, W) anomaly map."""
    if gaussian_filter is not None:
        return gaussian_filter(m, sigma=sigma)
    # Fallback with no new dependency: box blur of comparable width.
    k = max(3, int(2 * round(sigma) + 1))
    t = torch.from_numpy(m)[None, None].float()
    t = F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, stride=1)
    return t.squeeze().numpy()


class PaDiM:
    """Fit on normal images, score anything. No trainable parameters."""

    def __init__(self, n_dims: int = 100, seed: int = 1, device=None):
        self.n_dims = n_dims
        self.seed = seed
        self.device = device or torch.device("cpu")

        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)          # frozen: this is the whole point

        self.idx = None       # the random channel subset, drawn in fit()
        self.mean = None      # (n_patches, n_dims)
        self.inv_cov = None   # (n_patches, n_dims, n_dims)
        self.grid = None      # (h, w) of the patch grid

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float in [0,1] -> (B, C, h, w) multi-scale embedding."""
        if x.shape[1] == 1:                  # grayscale -> 3 channels for ResNet
            x = x.repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        m = self.backbone
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f1 = m.layer1(x)      # (B,  64, 32, 32) for 128x128 input
        f2 = m.layer2(f1)     # (B, 128, 16, 16)
        f3 = m.layer3(f2)     # (B, 256,  8,  8)
        size = f1.shape[-2:]
        return torch.cat([
            f1,
            F.interpolate(f2, size=size, mode="nearest"),
            F.interpolate(f3, size=size, mode="nearest"),
        ], dim=1)             # (B, 448, 32, 32)

    @torch.no_grad()
    def fit(self, x: np.ndarray, batch_size: int = 32) -> None:
        """x: (N, 3, H, W) float32 in [0,1], NORMAL images only."""
        if len(x) <= self.n_dims:
            raise ValueError(
                f"{len(x)} training images for n_dims={self.n_dims}: the per-patch "
                "covariance is singular. Use more images or a smaller --n-dims.")

        embeddings = []
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            embeddings.append(self._embed(batch).cpu())
        emb = torch.cat(embeddings)                  # (N, 448, h, w)

        # Draw the random channel subset ONCE, reproducibly. This is the only
        # stochastic part of PaDiM, so it is the only source of seed variance.
        rng = np.random.default_rng(self.seed)
        self.idx = torch.from_numpy(
            rng.choice(emb.shape[1], size=self.n_dims, replace=False))
        emb = emb[:, self.idx]                       # (N, n_dims, h, w)

        n, d, h, w = emb.shape
        self.grid = (h, w)
        emb = emb.reshape(n, d, h * w).numpy()       # (N, d, n_patches)

        self.mean = emb.mean(axis=0).T               # (n_patches, d)
        inv = np.empty((h * w, d, d), dtype=np.float64)
        eye = np.eye(d)
        for p in range(h * w):
            # rowvar=False wants (N, d); emb[:, :, p] is exactly that. Getting
            # this backwards computes an N x N covariance and nonsense scores.
            cov = np.cov(emb[:, :, p], rowvar=False) + COV_EPS * eye
            inv[p] = np.linalg.inv(cov)
        self.inv_cov = inv

    @torch.no_grad()
    def score(self, x: np.ndarray, batch_size: int = 32, sigma: float = 4.0,
              return_maps: bool = True):
        """Return (image_scores (N,), anomaly_maps (N, H, W) or None)."""
        if self.mean is None:
            raise RuntimeError("call fit() before score()")
        out_size = x.shape[-1]
        idx = self.idx.to(self.device)
        scores, maps = [], []

        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            emb = self._embed(batch)[:, idx].cpu().numpy()
            b, d, h, w = emb.shape
            flat = emb.reshape(b, d, h * w)

            dist = np.empty((b, h * w), dtype=np.float64)
            for p in range(h * w):
                delta = flat[:, :, p] - self.mean[p]              # (b, d)
                # Mahalanobis, batched: sum over d of (delta @ inv) * delta.
                # Clamp at 0 so float error cannot produce a NaN in sqrt.
                dist[:, p] = np.sqrt(np.maximum(
                    np.einsum("bi,ij,bj->b", delta, self.inv_cov[p], delta), 0.0))

            m = torch.from_numpy(dist.reshape(b, 1, h, w)).float()
            m = F.interpolate(m, size=out_size, mode="bilinear",
                              align_corners=False).squeeze(1).numpy()
            m = np.stack([_blur(mi, sigma) for mi in m])
            scores.append(m.reshape(len(m), -1).max(axis=1))
            if return_maps:
                maps.append(m.astype(np.float32))

        return (np.concatenate(scores),
                np.concatenate(maps) if return_maps else None)
