r"""Harmony integration (Korsunsky et al. 2019) via ``harmonypy``.

Harmony is iterative and has no closed-form "transform on new data"
step in its published form. Concretely, its fitted state consists of:

* L2-normalised cluster centroids ``Y`` (``d × K``),
* a soft-assignment bandwidth ``sigma`` (``K,``),
* a per-cluster ridge-regression batch correction ``W_batch``
  (``K × B × d``) that was applied to the training rows during the
  last ``moe_correct_ridge`` call.

Given those, the correction for a *new* sample reduces to a closed
form:

.. math::

    r_{ik}        &= \\mathrm{softmax}_k(-\\lVert \\hat{x}_i - Y_k
                       \\rVert^2 / \\sigma_k)

    \\hat{x}_i    &= x_i / \\lVert x_i \\rVert_2

    x_i^{\\text{corr}} &= x_i - \\sum_k r_{ik}\\, W_{\\text{batch}}^{\\,k,\\,
                                b(i)}

where :math:`b(i)` is sample :math:`i`'s batch index.  This mirrors
what Harmony does to training rows at its last iteration, but with
:math:`Y` and :math:`W_{\\text{batch}}` frozen at fit time, so test
rows cannot leak into the correction.

For the common case where ``fit`` and ``transform`` receive the same
rows (``fit_transform`` or simply re-``transform``), we short-circuit
to the exact ``harmonypy`` output so users see bit-identical behaviour
to the upstream library.

Built-in PCA preprocessing
--------------------------
Harmony was designed for low-dimensional dense embeddings (~20-50 PCA
components).  Applying it directly to a 6000-bin MALDI-TOF intensity
matrix degrades the soft-cluster assignment, because cosine distances
in the raw feature space are dominated by a handful of high-intensity
bins.  To sidestep that, :class:`Harmony` **always** fits a ``PCA``
on the training rows at ``fit`` time, runs Harmony in PCA space, and
maps the corrected result back to the original feature space via
``PCA.inverse_transform``.  The PCA itself is **frozen** at ``fit``
(like :math:`Y` and :math:`W_{\\text{batch}}`) and re-used verbatim
on unseen rows, so the closed-form transform stays leakage-safe by
construction.  Use the ``n_components`` argument to tune the size of
the PCA basis; ``None`` (the default) picks ``min(50, n_samples - 1,
n_features)``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .._base import BaseBatchCorrector
from .._utils import ArrayLike

_PCA_DEFAULT_N_COMPONENTS = 50

__all__ = ["Harmony"]


@contextlib.contextmanager
def _silence_harmonypy_logger():
    """Temporarily raise the ``harmonypy`` logger above INFO.

    ``harmonypy`` configures its own logger at module-import time with a
    DEBUG level and a StreamHandler attached, so setting ``verbose=False``
    on ``run_harmony`` alone does not suppress its progress messages.
    This context manager elevates the logger to WARNING for the duration
    of the call and restores the original state afterwards, leaving the
    user's broader logging configuration untouched.
    """
    logger = logging.getLogger("harmonypy")
    prev_level = logger.level
    prev_propagate = logger.propagate
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        yield
    finally:
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate


class Harmony(BaseBatchCorrector):
    """Sklearn-compatible Harmony with a closed-form ``transform``.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels.
    covariates : array-like, optional
        Additional categorical covariates forwarded to
        ``harmonypy.run_harmony`` via the ``vars_use`` mechanism.
    theta : float or list of float, default=2.0
        Diversity clustering penalty in Harmony.  Higher values
        encourage more aggressive batch mixing.
    max_iter : int, default=20
        Harmony iteration cap.
    nclust : int, optional
        Number of Harmony clusters.  When ``None``, ``harmonypy``'s
        default heuristic is used.
    n_components : int, optional
        Number of PCA components used for the built-in dimensionality-
        reduction stage.  ``None`` (the default) picks
        ``min(50, n_samples - 1, n_features)``.
    random_state : int, optional
        Seed forwarded to harmonypy **and** to the PCA stage.
    verbose : bool, default=True
        If ``False``, suppress ``harmonypy``'s own progress messages
        during ``fit``.  ``harmonypy`` configures its module-level
        logger eagerly at import time, so simply passing
        ``verbose=False`` to ``run_harmony`` is not enough; this flag
        also raises the ``harmonypy`` logger above INFO for the
        duration of the call.  The user's broader logging config is
        left untouched.
    scale_before_pca : bool, default=True
        If ``True`` (the default), fit a :class:`~sklearn.preprocessing.StandardScaler`
        on the training rows and apply it before the internal PCA.  This
        gives every feature equal weight in the principal components and
        matches the convention from single-cell genomics workflows
        (Seurat / Scanpy ``ScaleData`` followed by PCA), where Harmony
        was originally developed.  Without scaling, MALDI-TOF features
        with high intensity (a handful of intense peaks) dominate the
        PCs and Harmony's soft-cluster geometry inherits that bias.
        The scaler is **frozen** at ``fit`` (like the PCA, the centroids
        ``Y``, and the per-batch ridge ``W_batch``) and reused verbatim
        on unseen rows at ``transform`` time, with the corrected output
        un-scaled back to the original measurement scale.  Set to
        ``False`` to recover the pre-v0.2 behaviour (centred but not
        scaled before PCA).

    Attributes
    ----------
    batch_levels_ : np.ndarray
        Batch levels in the order used to encode ``Phi_moe`` columns.
    Y_ : np.ndarray of shape (n_components, K)
        L2-normalised cluster centroids frozen at fit time (in PCA
        space).
    sigma_ : np.ndarray of shape (K,)
        Soft-assignment bandwidths per cluster.
    W_batch_ : np.ndarray of shape (K, B, n_components)
        Per-cluster, per-batch ridge-regression correction vectors, in
        the same PCA space as ``Y_``.
    pca_ : sklearn.decomposition.PCA
        Fitted PCA, re-used verbatim on unseen rows at transform time.
    X_fit_ : np.ndarray
        Training feature matrix (original space) cached for
        same-matrix short-circuiting.
    n_clusters_ : int
        Number of clusters actually used by Harmony.
    n_iter_ : int
        Harmony iterations observed at fit time.

    Notes
    -----
    * The closed-form output on ``X_train`` is not quite bit-identical
      to ``harmonypy``'s iterative correction because Harmony's soft
      assignment is refined by a batch-diversity penalty during
      training (the ``update_R`` loop).  We deliberately use the raw
      cosine-softmax assignment at transform time since the diversity
      correction is a *training-time regulariser* tied to the training
      batch distribution and would not apply to a held-out sample.
      When ``transform`` is called on the exact training matrix we
      short-circuit to the harmonypy output so that ``fit_transform``
      matches upstream byte-for-byte.
    * Unseen batch levels at transform raise ``ValueError``, mirroring
      ``ComBat`` / ``Limma``.
    * Requires ``harmonypy``; it is a core dependency of
      MaldiBatchKit, range-pinned to ``>=0.2.0,<2``.  The 2.x line is
      a C++ rewrite that drops the private attributes our closed-form
      transform relies on (``_Phi_moe``, ``_lamb``, ``sigma``); we
      will revisit this constraint once those are re-exposed upstream.

    Examples
    --------
    >>> from maldibatchkit import Harmony
    >>> corrector = Harmony(batch=batches, random_state=0)
    >>> X_train_c = corrector.fit_transform(X_train)
    >>> X_test_c  = corrector.transform(X_test)   # closed-form on new rows
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        covariates: ArrayLike | None = None,
        theta: float | list[float] = 2.0,
        max_iter: int = 20,
        nclust: int | None = None,
        n_components: int | None = None,
        random_state: int | None = None,
        verbose: bool = True,
        scale_before_pca: bool = True,
    ) -> None:
        super().__init__(batch=batch)
        self.covariates = covariates
        self.theta = theta
        self.max_iter = max_iter
        self.nclust = nclust
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.scale_before_pca = scale_before_pca

    def _resolve_n_components(self, n_samples: int, n_features: int) -> int:
        """Pick a safe number of PCs for a given (n_samples, n_features)."""
        cap = max(1, min(n_samples - 1, n_features))
        if self.n_components is None:
            return min(_PCA_DEFAULT_N_COMPONENTS, cap)
        return int(min(self.n_components, cap))

    @staticmethod
    def _require_harmonypy():
        try:
            import harmonypy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Harmony requires the optional 'harmonypy' "
                "dependency. Install it with `pip install harmonypy` "
                "(it is pulled in automatically by `pip install maldibatchkit`)."
            ) from exc
        return harmonypy

    def _build_meta(
        self, idx: pd.Index, batch: npt.NDArray[Any]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Construct the harmonypy ``meta_data`` DataFrame + ``vars_use`` list."""
        meta = pd.DataFrame(
            {"batch": np.asarray(batch, dtype=object).astype(str)},
            index=idx,
        )
        vars_use = ["batch"]
        if self.covariates is not None:
            cov = self.covariates
            if isinstance(cov, pd.Series):
                cov = cov.to_frame()
            if not isinstance(cov, pd.DataFrame):
                cov = pd.DataFrame(np.asarray(cov), index=idx)
            cov_aligned = cov.loc[idx].copy().astype(str)
            cov_aligned.columns = [f"cov_{i}" for i in range(cov_aligned.shape[1])]
            meta = pd.concat([meta, cov_aligned], axis=1)
            vars_use.extend(list(cov_aligned.columns))
        return meta, vars_use

    @staticmethod
    def _l2_normalise_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, eps, None)
        return X / norms

    def _softmax_assignment(self, X: np.ndarray) -> np.ndarray:
        """Compute the frozen-centroid soft assignment R (N, K)."""
        Z_cos = self._l2_normalise_rows(X)
        dist = 2.0 * (1.0 - Z_cos @ self.Y_)  # (N, K)
        scaled = -dist / self.sigma_[None, :]
        scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
        R = np.exp(scaled)
        R = R / np.clip(R.sum(axis=1, keepdims=True), 1e-12, None)
        return R

    @staticmethod
    def _extract_W_batch(harmony_out, batch_levels: np.ndarray) -> np.ndarray:
        """Recompute the final-iteration per-cluster per-batch ridge W.

        Harmony's ``Phi_moe`` has shape ``(1 + ΣL_v, N)`` where ``L_v``
        is the number of levels of each variable in ``vars_use``.  The
        batch indicator occupies **the first** level block (rows
        ``1 : 1 + n_batches``), because ``_build_meta`` always places
        ``"batch"`` first in ``vars_use``.  Additional covariates
        contribute further level blocks that are *protected* in the
        training correction but are not reapplied to unseen rows at
        transform time - the closed-form path removes batch only.

        Returns an array of shape ``(K, n_batches, d)`` such that
        ``W_batch[k, b, :]`` is the per-cluster, per-batch coefficient
        subtracted for training samples that belong to batch ``b``.
        """
        Z_orig = np.asarray(harmony_out._Z_orig.cpu().numpy(), dtype=float)  # (d, N)
        Phi_moe = np.asarray(
            harmony_out._Phi_moe.cpu().numpy(), dtype=float
        )  # (rows, N)
        R = np.asarray(harmony_out._R.cpu().numpy(), dtype=float)  # (K, N)
        lamb = np.asarray(harmony_out._lamb.cpu().numpy(), dtype=float)  # (rows,)

        K, _ = R.shape
        d = Z_orig.shape[0]
        n_batches = int(len(batch_levels))

        W_batch = np.zeros((K, n_batches, d), dtype=float)
        batch_slice = slice(1, 1 + n_batches)  # skip intercept, take batch block
        for k in range(K):
            Phi_Rk = Phi_moe * R[k]  # (rows, N)
            cov = Phi_Rk @ Phi_moe.T + np.diag(lamb)  # (rows, rows)
            inv_cov = np.linalg.inv(cov)
            Z_tmp = Z_orig * R[k]  # (d, N)
            rhs = Phi_moe @ Z_tmp.T  # (rows, d)
            W_k = inv_cov @ rhs  # (rows, d)
            W_k[0, :] = 0.0  # keep the intercept
            W_batch[k] = W_k[batch_slice, :]
        return W_batch

    def _extract_batch_levels(self, harmony_out, batch: npt.NDArray[Any]) -> np.ndarray:
        """Extract the batch-level ordering implied by Phi_moe's columns.

        ``harmonypy`` uses ``pd.factorize(meta[vars_use[0]])`` under the
        hood, so the order is the first-occurrence order in the meta
        DataFrame we provided.
        """
        seen: list[str] = []
        for b in batch.tolist():
            s = str(b)
            if s not in seen:
                seen.append(s)
        return np.asarray(seen, dtype=object)

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        harmonypy = self._require_harmonypy()
        if self.random_state is not None:
            np.random.seed(self.random_state)

        meta, vars_use = self._build_meta(X_df.index, batch)
        X_orig = X_df.to_numpy(dtype=float)

        # Optional StandardScaler before PCA: gives every feature equal
        # weight in the principal components, matching the Seurat /
        # Scanpy ``ScaleData -> PCA`` convention from single-cell
        # genomics where Harmony was first developed.  The scaler is
        # frozen at fit time and reused on unseen rows at transform.
        if self.scale_before_pca:
            self.scaler_ = StandardScaler().fit(X_orig)
            X_scaled = self.scaler_.transform(X_orig)
        else:
            self.scaler_ = None
            X_scaled = X_orig

        # PCA preprocessing is mandatory and fit on training rows only.
        # It is then frozen and re-used verbatim by ``transform`` so no
        # information leaks from test rows back into the basis.
        n_pcs = self._resolve_n_components(X_scaled.shape[0], X_scaled.shape[1])
        self.pca_ = PCA(n_components=n_pcs, random_state=self.random_state)
        X_work = self.pca_.fit_transform(X_scaled)

        kwargs: dict[str, Any] = {
            "vars_use": vars_use,
            "theta": self.theta,
            "max_iter_harmony": self.max_iter,
            "verbose": self.verbose,
        }
        if self.nclust is not None:
            kwargs["nclust"] = self.nclust

        log_ctx = (
            _silence_harmonypy_logger()
            if not self.verbose
            else contextlib.nullcontext()
        )
        with log_ctx:
            harmony_out = harmonypy.run_harmony(X_work, meta, **kwargs)

        self.batch_levels_ = self._extract_batch_levels(harmony_out, batch)
        # Y_ / W_batch_ live in PCA space.  The closed-form transform
        # projects new rows into the same space before applying them.
        self.Y_ = harmony_out.Y  # (n_components, K)
        self.sigma_ = harmony_out.sigma.astype(float)  # (K,)
        self.W_batch_ = self._extract_W_batch(
            harmony_out, self.batch_levels_
        )  # (K, B, n_components)
        self.n_clusters_ = int(self.Y_.shape[1])
        self.n_iter_ = int(len(getattr(harmony_out, "kmeans_rounds", []) or [0]))

        # Same-matrix fast-path cache in the original feature space -
        # map harmonypy's Z_corr back through the fitted PCA so users
        # who call ``transform(X_train)`` (or ``fit_transform``) see
        # ``(n_samples, n_features)``.  When the scaler is in use we
        # also un-standardise so the cached output is on the original
        # measurement scale, matching the closed-form transform path.
        Z_corr_work = np.asarray(harmony_out.Z_corr)  # (N, n_components)
        Z_corr_scaled = self.pca_.inverse_transform(Z_corr_work)
        if self.scaler_ is not None:
            Z_corr_orig = self.scaler_.inverse_transform(Z_corr_scaled)
        else:
            Z_corr_orig = Z_corr_scaled
        self.X_fit_ = X_orig
        self._corrected_cache_ = Z_corr_orig  # (N, n_features)

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        X_orig = X_df.to_numpy(dtype=float)

        # Fast path: caller transforms the exact matrix we fit on ->
        # return the cached correction byte-for-byte.
        if X_orig.shape == self.X_fit_.shape and np.array_equal(X_orig, self.X_fit_):
            return pd.DataFrame(
                self._corrected_cache_,
                index=X_df.index,
                columns=X_df.columns,
            )

        # Closed-form correction on new rows --------------------------
        batch_str = np.asarray([str(b) for b in batch])
        known = {lvl: i for i, lvl in enumerate(self.batch_levels_)}
        unseen = sorted(set(batch_str) - set(known))
        if unseen:
            raise ValueError(
                f"Unseen batch level(s) at transform: {unseen}. "
                f"Harmony's closed-form correction is only defined for "
                f"batches that were present at fit time."
            )
        batch_idx = np.asarray([known[b] for b in batch_str], dtype=int)

        # Project into PCA space using the fitted basis - no data from
        # ``X_orig`` flows back into the scaler / PCA at fit time, so
        # the train/test split stays clean.
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_orig)
        else:
            X_scaled = X_orig
        X_work = self.pca_.transform(X_scaled)

        R_new = self._softmax_assignment(X_work)  # (N, K)

        # W_batch_ shape (K, B, n_components); pull the batch slice per
        # sample -> (N, K, n_components); weight by R_new and sum over K.
        W_selected = self.W_batch_[:, batch_idx, :].transpose(1, 0, 2)
        correction = np.einsum("nk,nkd->nd", R_new, W_selected)

        Z_corr_work = X_work - correction
        Z_corr_scaled = self.pca_.inverse_transform(Z_corr_work)
        if self.scaler_ is not None:
            Z_corr_orig = self.scaler_.inverse_transform(Z_corr_scaled)
        else:
            Z_corr_orig = Z_corr_scaled
        return pd.DataFrame(Z_corr_orig, index=X_df.index, columns=X_df.columns)
