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

# Channels in the multi-scale embedding: resnet18 layer1 + layer2 + layer3.
# Fixed by the backbone, so the channel subset can be drawn before embedding
# anything - which is what lets the server choose it for the whole federation.
EMBED_CHANNELS = 64 + 128 + 256  # = 448


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

    def _draw_channels(self, idx: np.ndarray | None) -> None:
        """Set self.idx: the given subset, or a fresh one drawn from self.seed.

        Drawing it from the seed is the only stochastic part of PaDiM, so it is
        the only source of seed variance. In the federated setting the server
        passes idx in, because every client MUST use the same subset (channel i
        of one client is averaged with channel i of another).
        """
        if idx is None:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(EMBED_CHANNELS, size=self.n_dims, replace=False)
        self.idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))

    @torch.no_grad()
    def _embed_all(self, x: np.ndarray, batch_size: int) -> torch.Tensor:
        """(N, 3, H, W) -> (N, n_dims, h, w) on the CPU, batched to bound memory.

        Selects self.idx inside the loop, so only n_dims channels are ever kept:
        at n_dims=100 that is 4.5x less memory than holding all 448.
        Requires self.idx to be set already (_draw_channels).
        """
        idx = self.idx.to(self.device)
        out = []
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            out.append(self._embed(batch)[:, idx].cpu())
        return torch.cat(out)                        # (N, n_dims, h, w)

    @torch.no_grad()
    def fit(self, x: np.ndarray, batch_size: int = 32,
            idx: np.ndarray | None = None) -> None:
        """x: (N, 3, H, W) float32 in [0,1], NORMAL images only.

        idx: the channel subset to use. None (default) draws it from self.seed,
        as the paper does. Pass it in the federated setting, where the server
        chooses one subset for the whole federation (see experiments/padim_federated.py).
        """
        if len(x) <= self.n_dims:
            raise ValueError(
                f"{len(x)} training images for n_dims={self.n_dims}: the per-patch "
                "covariance is singular. Use more images or a smaller --n-dims.")

        self._draw_channels(idx)
        emb = self._embed_all(x, batch_size)         # (N, n_dims, h, w)

        n, d, h, w = emb.shape
        self.grid = (h, w)
        # float64: the mean and covariance are sums over N images, and in float32
        # the accumulation costs ~1e-6 of precision. Cheap here, and it makes this
        # path agree exactly with the federated aggregation in padim_federated.py.
        emb = emb.reshape(n, d, h * w).double().numpy()   # (N, d, n_patches)

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
    def local_statistics(self, x: np.ndarray, idx: np.ndarray,
                         batch_size: int = 32) -> dict:
        """Sufficient statistics of one client's NORMAL images, for federation.

        Returns {"n": int, "S": (n_patches, d), "Q": (n_patches, d, d), "grid": (h, w)}
        where S = sum of embeddings and Q = sum of outer products, per patch
        position. Mean and covariance are reconstructible EXACTLY from these sums,
        so the server can build the same model it would have built on the pooled
        images. No image, and no per-image quantity, leaves the client.

        Deliberately does NOT invert anything: inv(mean) != mean(inv), so the
        server must sum the second moments and invert once (trap 2 of the guide).
        """
        if len(x) == 0:
            raise ValueError("no training images: nothing to compute statistics from")
        self._draw_channels(idx)                     # server-chosen, never drawn here

        # Accumulated batch by batch, never holding the whole embedding: client
        # memory then depends on batch_size, not on how many images the client
        # owns. S and Q are sums, so batching them is exact.
        #
        # float64: Q accumulates sums of squares, and the cancellation in
        # Q/n - mu mu^T can cost several significant digits in float32 (and even
        # yield a non-PSD matrix). COV_EPS hides a lot; do not rely on it to hide
        # a numerical bug.
        d = self.n_dims
        n_seen, grid, S, Q = 0, None, None, None
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            emb = self._embed(batch)[:, self.idx.to(self.device)].cpu()
            b, _, h, w = emb.shape
            if S is None:
                grid = (h, w)
                S = np.zeros((h * w, d), dtype=np.float64)
                Q = np.zeros((h * w, d, d), dtype=np.float64)
            elif grid != (h, w):
                raise ValueError(
                    f"batch {i // batch_size} has grid {(h, w)}, expected {grid}: "
                    "the client's images are not all the same size.")
            e = emb.reshape(b, d, h * w).permute(2, 0, 1).double().numpy()  # (p, b, d)
            S += e.sum(axis=1)
            Q += np.einsum("pnd,pne->pde", e, e)
            n_seen += b

        return {"n": int(n_seen), "S": S, "Q": Q, "grid": grid}

    def load_statistics(self, mean: np.ndarray, inv_cov: np.ndarray,
                        idx: np.ndarray, grid: tuple[int, int]) -> None:
        """Make this instance scorable from an aggregated model (no fit, no images).

        The server side of federation: score() only reads mean, inv_cov and idx,
        so a model assembled from clients' statistics scores exactly like a fitted one.
        """
        self.mean = np.asarray(mean)
        self.inv_cov = np.asarray(inv_cov)
        self.idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        self.grid = tuple(grid)

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
