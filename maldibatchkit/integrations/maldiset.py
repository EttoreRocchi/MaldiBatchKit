"""``MaldiSet`` <-> MaldiBatchKit bridge."""

from __future__ import annotations

import inspect
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - type-checking imports only
    from maldiamrkit import MaldiSet

__all__ = ["MaldiSetAdapter"]


def _require_maldiset():
    try:
        from maldiamrkit import MaldiSet
    except ImportError as exc:
        raise ImportError(
            "MaldiSetAdapter requires `maldiamrkit` (normally installed "
            "alongside MaldiBatchKit). Reinstall with "
            "`pip install -U maldibatchkit` or `pip install maldiamrkit`."
        ) from exc
    return MaldiSet


class MaldiSetAdapter:
    """Bridge :class:`maldiamrkit.MaldiSet` and MaldiBatchKit correctors.

    The adapter does three things:

    1. Extracts the feature matrix ``ds.X`` and any metadata columns
       (batch / species / quality) from a ``MaldiSet``.
    2. Delegates to a MaldiBatchKit corrector (or any sklearn-compatible
       transformer with the right constructor signature) to produce a
       corrected matrix.
    3. Returns a **new** ``MaldiSet`` whose ``X`` property yields the
       corrected matrix, leaving the original dataset untouched.

    Batch / covariate slicing follows the rest of the package: pass the
    metadata column names at construction time; the adapter pulls the
    aligned series from ``ds.meta`` itself so users do not have to
    rebuild arrays manually.

    Parameters
    ----------
    batch_column : str
        Column in ``ds.meta`` with the batch labels.
    species_column : str, optional
        Column in ``ds.meta`` with species labels (used when the chosen
        corrector needs a species covariate).
    quality_column : str, optional
        Column in ``ds.meta`` with per-sample quality scores (used by
        :class:`maldibatchkit.QualityWeightedComBat`).

    Examples
    --------
    >>> from maldiamrkit import MaldiSet
    >>> from maldibatchkit.integrations import MaldiSetAdapter
    >>> from maldibatchkit import SpeciesAwareComBat
    >>> adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
    >>> corrected_ds = adapter.correct(ds, SpeciesAwareComBat)
    >>> corrected_ds.X.head()   # corrected feature matrix
    """

    def __init__(
        self,
        *,
        batch_column: str,
        species_column: str | None = None,
        quality_column: str | None = None,
    ) -> None:
        self.batch_column = batch_column
        self.species_column = species_column
        self.quality_column = quality_column

    def extract(self, ds: MaldiSet) -> dict[str, Any]:
        """Pull ``X``, batch, species, quality from a ``MaldiSet``.

        Returns
        -------
        dict
            Dictionary with keys ``X`` (DataFrame), ``batch``,
            ``species`` (or ``None``), ``quality`` (or ``None``). The
            series are aligned to ``X.index``.
        """
        _require_maldiset()
        X = ds.X
        idx = X.index
        meta = ds.meta.loc[idx]
        if self.batch_column not in meta.columns:
            raise KeyError(
                f"batch_column={self.batch_column!r} not found in MaldiSet metadata."
            )
        batch = meta[self.batch_column]
        species = (
            meta[self.species_column]
            if self.species_column and self.species_column in meta.columns
            else None
        )
        quality = (
            meta[self.quality_column]
            if self.quality_column and self.quality_column in meta.columns
            else None
        )
        return {"X": X, "batch": batch, "species": species, "quality": quality}

    def correct(
        self,
        ds: MaldiSet,
        transformer_cls,
        *,
        transformer_kwargs: dict[str, Any] | None = None,
    ) -> MaldiSet:
        """Run ``transformer_cls(batch=..., ...)`` and return a new MaldiSet.

        Parameters
        ----------
        ds : MaldiSet
            Source dataset.
        transformer_cls : type
            A MaldiBatchKit corrector class (or any transformer whose
            constructor takes ``batch=`` and, optionally, a
            species-style protected-covariate argument named
            ``species`` / ``discrete_covariates`` / ``design`` /
            ``covariates`` and/or a ``quality=`` argument).
        transformer_kwargs : dict, optional
            Extra keyword arguments forwarded to ``transformer_cls``.

        Returns
        -------
        MaldiSet
            Shallow-copied dataset with its ``_X_cache`` replaced by the
            corrected feature matrix. The ``spectra`` list is the same
            object in the returned ``MaldiSet``; labels / metadata are
            unchanged.
        """
        _require_maldiset()
        extracted = self.extract(ds)
        X = extracted["X"]
        batch = extracted["batch"]
        species = extracted["species"]
        quality = extracted["quality"]

        kwargs: dict[str, Any] = {"batch": batch}
        try:
            init_params = set(inspect.signature(transformer_cls.__init__).parameters)
        except (TypeError, ValueError):  # pragma: no cover - exotic subclasses
            init_params = set()

        # Route species metadata to a *categorical* protected-covariate
        # slot, in decreasing order of specificity: dedicated
        # ``species=``, then ComBat / Fortin's ``discrete_covariates=``,
        # then Harmony's ``covariates=``.  We intentionally do NOT auto-
        # route to Limma's ``design=`` (that slot is a numeric design
        # matrix, not a label); Limma users who want to protect species
        # should pass e.g. ``transformer_kwargs={"design": species_dummies}``.
        if species is not None:
            for name in ("species", "discrete_covariates", "covariates"):
                if name in init_params:
                    kwargs[name] = species
                    break
        if quality is not None and "quality" in init_params:
            kwargs["quality"] = quality
        if transformer_kwargs:
            kwargs.update(transformer_kwargs)

        transformer = transformer_cls(**kwargs)
        X_corrected = transformer.fit_transform(X)
        if not isinstance(X_corrected, pd.DataFrame):
            X_corrected = pd.DataFrame(
                np.asarray(X_corrected), index=X.index, columns=X.columns
            )

        new_ds = copy(ds)
        # MaldiSet lazily rebuilds X from spectra unless _X_cache is set;
        # inject the corrected matrix there so every downstream property
        # (``.y``, ``.filter()``...) keeps working on the corrected data.
        new_ds._X_cache = X_corrected
        new_ds.meta = ds.meta.loc[X_corrected.index].copy()
        return new_ds
