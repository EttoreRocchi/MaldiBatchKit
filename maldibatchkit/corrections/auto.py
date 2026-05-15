"""sklearn-compatible meta-corrector with a swappable ``method`` hyperparameter.

:class:`AutoCorrector` is the entry point for *selecting* a corrector
instead of picking one by hand. Wrap it in
``sklearn.model_selection.GridSearchCV`` with a ``param_grid`` that
sweeps ``method`` across corrector families and let the downstream
classifier score (typically AUROC) decide.

Resolution rules
----------------
``method`` accepts either a registered string alias (see
:data:`METHOD_REGISTRY`) or a :class:`BaseBatchCorrector` *subclass*.
Constructor kwargs are forwarded to the inner method only when its
``__init__`` actually accepts them (resolved via
:func:`inspect.signature`); unrecognised kwargs are dropped silently.
That matters because the same ``param_grid`` typically targets several
methods, only some of which accept e.g. ``quality`` or ``reference_batch``.

Covariate handling
------------------
``AutoCorrector`` mirrors :class:`combatlearn.ComBat`'s split between
``discrete_covariates`` and ``continuous_covariates`` so each method
receives the kwarg name it understands:

* ComBat (Fortin / Chen / Johnson) - both forwarded verbatim;
* :class:`~maldibatchkit.Harmony` - ``discrete_covariates`` is forwarded
  as ``covariates`` (Harmony's categorical covariates); ``continuous_covariates``
  is dropped with a warning;
* :class:`~maldibatchkit.Limma` - ``discrete_covariates`` is forwarded
  as ``design`` (Limma's design of interest); ``continuous_covariates``
  is dropped with a warning;
* :class:`~maldibatchkit.SpeciesAwareComBat` - needs ``species=`` directly
  (a biological covariate is not a generic discrete covariate);
* :class:`~maldibatchkit.QualityWeightedComBat` - needs ``quality=`` directly;
* baselines and :class:`~maldibatchkit.NoOpCorrector` - covariates dropped.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .._base import BaseBatchCorrector
from .._utils import ArrayLike
from .baselines import MedianCentering, ReferenceScaling, ZScorePerBatch
from .combat import ComBat, SpeciesAwareComBat
from .harmony import Harmony
from .limma import Limma
from .maldi import BatchAwareWarping
from .noop import NoOpCorrector
from .quality_weighted import QualityWeightedComBat

__all__ = ["AutoCorrector", "METHOD_REGISTRY"]


def _combat_johnson(**kw: Any) -> ComBat:
    return ComBat(method="johnson", **kw)


def _combat_fortin(**kw: Any) -> ComBat:
    return ComBat(method="fortin", **kw)


def _combat_chen(**kw: Any) -> ComBat:
    return ComBat(method="chen", **kw)


METHOD_REGISTRY: dict[str, Any] = {
    "combat": _combat_fortin,
    "combat-johnson": _combat_johnson,
    "combat-fortin": _combat_fortin,
    "combat-chen": _combat_chen,
    "limma": Limma,
    "harmony": Harmony,
    "qw-combat": QualityWeightedComBat,
    "species-combat": SpeciesAwareComBat,
    "batch-warping": BatchAwareWarping,
    "median": MedianCentering,
    "zscore": ZScorePerBatch,
    "reference": ReferenceScaling,
    "noop": NoOpCorrector,
}


_COVARIATE_ALIASES: dict[str, tuple[str, str | None]] = {
    "combat": ("discrete_covariates", "continuous_covariates"),
    "combat-johnson": ("discrete_covariates", "continuous_covariates"),
    "combat-fortin": ("discrete_covariates", "continuous_covariates"),
    "combat-chen": ("discrete_covariates", "continuous_covariates"),
    "species-combat": ("discrete_covariates", "continuous_covariates"),
    "harmony": ("covariates", None),
    "limma": ("design", None),
}


def _resolve_method_key(method: Any) -> tuple[str | None, Any]:
    """Return ``(alias, builder)`` for a string or class ``method``."""
    if isinstance(method, str):
        key = method.lower()
        if key not in METHOD_REGISTRY:
            raise ValueError(
                f"Unknown method alias {method!r}. Registered aliases: "
                f"{sorted(METHOD_REGISTRY)}."
            )
        return key, METHOD_REGISTRY[key]
    if inspect.isclass(method) and issubclass(method, BaseBatchCorrector):
        return None, method
    raise TypeError(
        "method must be a registered string alias or a BaseBatchCorrector "
        f"subclass; got {type(method).__name__}."
    )


def _accepted_kwargs(builder: Any) -> set[str]:
    """Return the kwargs ``builder.__init__`` accepts (excluding self/batch)."""
    target = (
        builder
        if inspect.isclass(builder)
        else builder.__wrapped__
        if hasattr(builder, "__wrapped__")
        else builder
    )
    try:
        sig = inspect.signature(target if inspect.isclass(target) else target)
    except (TypeError, ValueError):
        return set()
    accepted: set[str] = set()
    has_var_keyword = False
    for name, p in sig.parameters.items():
        if name in ("self", "batch"):
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        accepted.add(name)
    if has_var_keyword and inspect.isclass(builder) and issubclass(builder, ComBat):
        # ComBat factory wrappers swallow extra kw; check the real class.
        try:
            sig_real = inspect.signature(ComBat.__init__)
            for name, p in sig_real.parameters.items():
                if name in ("self", "batch"):
                    continue
                if p.kind in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    continue
                accepted.add(name)
        except (TypeError, ValueError):
            pass
    return accepted


def _accepted_for_factory(alias: str | None, builder: Any) -> set[str]:
    """Return accepted kwargs, walking through alias factory wrappers when needed."""
    if alias in {"combat", "combat-johnson", "combat-fortin", "combat-chen"}:
        try:
            sig = inspect.signature(ComBat.__init__)
        except (TypeError, ValueError):
            return _accepted_kwargs(builder)
        accepted = set()
        for name, p in sig.parameters.items():
            if name in ("self", "batch"):
                continue
            if p.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            accepted.add(name)
        # Drop ``method`` because the factory pins it.
        accepted.discard("method")
        return accepted
    return _accepted_kwargs(builder)


class AutoCorrector(BaseBatchCorrector):
    """Meta-corrector with a swappable ``method`` hyperparameter.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels.
    method : str or BaseBatchCorrector subclass, default='combat-fortin'
        Either a registered string alias (see :data:`METHOD_REGISTRY`)
        or a :class:`BaseBatchCorrector` subclass.
    discrete_covariates : array-like, optional
        Categorical covariates. Forwarded under the kwarg the inner
        method actually understands (see module docstring).
    continuous_covariates : array-like, optional
        Continuous covariates. Forwarded to ComBat variants; dropped
        with a warning for methods that don't accept them.
    quality : array-like, optional
        Per-sample quality weights. Only used by
        :class:`~maldibatchkit.QualityWeightedComBat`.
    species : array-like, optional
        Species labels. Only used by
        :class:`~maldibatchkit.SpeciesAwareComBat`.
    reference_batch : Any, optional
        Reference batch level. Forwarded where the inner method accepts it.
    method_kwargs : dict, optional
        Extra kwargs merged into the inner method's ``__init__`` call.
        ``method_kwargs`` entries override any conflicting argument
        synthesised from the named parameters above; entries whose key
        is not accepted by the inner ``__init__`` are dropped silently.

    Attributes
    ----------
    inner_ : BaseBatchCorrector
        The instantiated inner corrector, fitted to the training rows.
    feature_names_in_, n_features_in_
        Forwarded from the inner corrector.

    Notes
    -----
    Fitted attributes of the inner corrector (e.g. ``gamma_star_`` on a
    ComBat inner) are reachable on the ``AutoCorrector`` itself via
    attribute fall-through (``__getattr__``), so ``check_is_fitted`` and
    downstream inspection work without further glue.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import Pipeline
    >>> from maldibatchkit import AutoCorrector
    >>> pipe = Pipeline([
    ...     ("correct", AutoCorrector(batch=b)),
    ...     ("clf", LogisticRegression()),
    ... ])
    >>> grid = GridSearchCV(
    ...     pipe,
    ...     param_grid={"correct__method": ["noop", "combat-fortin", "harmony"]},
    ...     scoring="roc_auc",
    ... )
    >>> grid.fit(X, y)  # doctest: +SKIP
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        method: Any = "combat-fortin",
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        quality: ArrayLike | None = None,
        species: ArrayLike | None = None,
        reference_batch: Any | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(batch=batch)
        self.method = method
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.quality = quality
        self.species = species
        self.reference_batch = reference_batch
        self.method_kwargs = method_kwargs

    def _build_inner(self) -> BaseBatchCorrector:
        alias, builder = _resolve_method_key(self.method)
        accepted = _accepted_for_factory(alias, builder)

        candidate: dict[str, Any] = {}

        # Map our generic covariate kwargs onto whatever the inner accepts.
        disc_alias, cont_alias = _COVARIATE_ALIASES.get(
            alias, ("discrete_covariates", "continuous_covariates")
        )
        if self.discrete_covariates is not None:
            if disc_alias and disc_alias in accepted:
                candidate[disc_alias] = self.discrete_covariates
            else:
                warnings.warn(
                    f"method={self.method!r} does not accept discrete "
                    f"covariates; ignoring `discrete_covariates`.",
                    stacklevel=2,
                )
        if self.continuous_covariates is not None:
            if cont_alias and cont_alias in accepted:
                candidate[cont_alias] = self.continuous_covariates
            else:
                warnings.warn(
                    f"method={self.method!r} does not accept continuous "
                    f"covariates; ignoring `continuous_covariates`.",
                    stacklevel=2,
                )
        if self.quality is not None and "quality" in accepted:
            candidate["quality"] = self.quality
        if self.species is not None and "species" in accepted:
            candidate["species"] = self.species
        if self.reference_batch is not None and "reference_batch" in accepted:
            candidate["reference_batch"] = self.reference_batch

        if self.method_kwargs:
            for k, v in self.method_kwargs.items():
                if k in accepted:
                    candidate[k] = v

        return builder(batch=self.batch, **candidate)

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        inner = self._build_inner()
        inner.fit(X_df)
        self.inner_ = inner
        self.feature_names_in_ = np.asarray(
            getattr(inner, "feature_names_in_", X_df.columns), dtype=object
        )

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        out = self.inner_.transform(X_df)
        if isinstance(out, pd.DataFrame):
            return out
        return pd.DataFrame(np.asarray(out), index=X_df.index, columns=X_df.columns)

    def __getattr__(self, name: str) -> Any:
        # Fall through to the fitted inner so consumers can read e.g.
        # ``gamma_star_`` directly off the AutoCorrector. Trailing
        # underscores filter to fitted attributes.
        if name.startswith("_") or not name.endswith("_"):
            raise AttributeError(name)
        inner = self.__dict__.get("inner_")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)
