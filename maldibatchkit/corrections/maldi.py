"""MALDI-specific batch corrections that leverage ``maldiamrkit`` internals."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .._base import BaseBatchCorrector
from .._utils import ArrayLike

__all__ = ["BatchAwareWarping"]


def _try_import_maldiamrkit():
    try:
        import maldiamrkit  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "This transformer requires 'maldiamrkit' (normally installed "
            "alongside MaldiBatchKit). Reinstall with "
            "`pip install -U maldibatchkit` or "
            "`pip install maldiamrkit>=0.13.0`."
        ) from exc
    return maldiamrkit


class BatchAwareWarping(BaseBatchCorrector):
    r"""Per-batch m/z warping to a single global reference.

    This transformer reuses :class:`maldiamrkit.alignment.Warping` under
    the hood, but fits one warper **per batch**. At ``fit`` time we
    learn a global reference spectrum from all training rows, then fit a
    dedicated ``Warping`` instance per batch with that shared reference.
    At ``transform`` time each sample is routed through the warper of
    its batch.

    Why this matters
    ----------------
    Mass-calibration drift is usually a per-acquisition-session
    phenomenon. Running a single global ``Warping`` can either be too
    aggressive (squashing real biological peaks in well-calibrated
    batches) or too conservative (leaving large shifts in a miscalibrated
    batch). Per-batch warpers share the reference but learn a
    batch-specific transformation.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels.
    reference : str, default="median"
        How to build the global reference.
        ``"median"`` uses the median spectrum across all training rows.
        ``"batch_mean"`` uses the mean of per-batch medians (reduces
        bias towards large batches).
    method : str, default="shift"
        Warping method passed through to
        ``maldiamrkit.alignment.Warping``.
    n_segments : int, default=5
        Piecewise warping segments.
    max_shift : int, default=50
        Maximum shift in bins.
    dtw_radius : int, default=10
        DTW radius constraint.
    smooth_sigma : float, default=2.0
        Gaussian smoothing for piecewise shifts.
    n_jobs : int, default=1
        Parallel jobs forwarded to ``Warping``.

    Attributes
    ----------
    reference\_ : np.ndarray
        The global reference spectrum.
    warpers\_ : dict[Any, Warping]
        One fitted ``Warping`` per batch level.

    Notes
    -----
    For batches that do not appear at ``fit`` time but show up during
    ``transform``, we fall back to a global warper trained on all
    training rows (stored as :attr:`warpers_` under the ``"__global__"``
    key). This keeps the transformer defined on test folds without
    peeking at test data.

    Examples
    --------
    >>> from maldibatchkit import BatchAwareWarping
    >>> warper = BatchAwareWarping(batch=batches, method="piecewise")
    >>> X_aligned = warper.fit_transform(X)
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        reference: str = "median",
        method: str = "shift",
        n_segments: int = 5,
        max_shift: int = 50,
        dtw_radius: int = 10,
        smooth_sigma: float = 2.0,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(batch=batch)
        self.reference = reference
        self.method = method
        self.n_segments = n_segments
        self.max_shift = max_shift
        self.dtw_radius = dtw_radius
        self.smooth_sigma = smooth_sigma
        self.n_jobs = n_jobs

    def _make_warper(self):
        _try_import_maldiamrkit()
        from maldiamrkit.alignment import Warping  # type: ignore[import-not-found]

        return Warping(
            reference="median",
            method=self.method,
            n_segments=self.n_segments,
            max_shift=self.max_shift,
            dtw_radius=self.dtw_radius,
            smooth_sigma=self.smooth_sigma,
            n_jobs=self.n_jobs,
        )

    def _build_reference(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> np.ndarray:
        if self.reference == "median":
            return X_df.median(axis=0).to_numpy()
        if self.reference == "batch_mean":
            per_batch = []
            for lvl in np.unique(batch):
                per_batch.append(X_df.loc[batch == lvl].median(axis=0).to_numpy())
            return np.mean(per_batch, axis=0)
        raise ValueError(
            f"Unknown reference={self.reference!r}; expected 'median' or 'batch_mean'."
        )

    def _fit_warper_on(self, df: pd.DataFrame, reference: np.ndarray):
        w = self._make_warper()
        w.fit(df)
        w.ref_spec_ = reference  # swap in the shared global reference
        return w

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        self.reference_ = self._build_reference(X_df, batch)
        warpers: dict[Any, Any] = {}
        for lvl in np.unique(batch):
            df = X_df.loc[batch == lvl]
            warpers[lvl] = self._fit_warper_on(df, self.reference_)
        warpers["__global__"] = self._fit_warper_on(X_df, self.reference_)
        self.warpers_ = warpers

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        out = X_df.copy().astype(float)
        for lvl in np.unique(batch):
            mask = batch == lvl
            chunk = X_df.loc[mask]
            warper = self.warpers_.get(lvl, self.warpers_["__global__"])
            out.loc[mask] = warper.transform(chunk).to_numpy()
        return out
