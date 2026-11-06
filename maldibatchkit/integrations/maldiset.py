"""``MaldiSet`` <-> MaldiBatchKit bridge."""

from __future__ import annotations

import functools
import inspect
import warnings
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - type-checking imports only
    from maldiamrkit import MaldiSet

__all__ = ["MaldiSetAdapter"]

_COVARIATE_KWARGS = {"species", "discrete_covariates", "covariates", "design"}


@functools.cache
def _corrected_class(base: type) -> type:
    """Subclass of ``base`` whose ``filter`` returns subsets of the corrected X."""

    def filter(self: Any, *filters: Any) -> Any:
        out = base.filter(self, *filters)
        out.__class__ = type(self)
        out._X_cache = self._X_cache.loc[self._X_cache.index.isin(out.meta.index)]
        out.meta = out.meta.loc[out._X_cache.index].rename_axis(out.meta.index.name)
        return out

    def __reduce__(self: Any) -> tuple:
        return _restore_corrected, (base, self.__dict__)

    return type(
        f"Corrected{base.__name__}",
        (base,),
        {
            "filter": filter,
            "__reduce__": __reduce__,
            "_corrected_base": base,
            "__module__": __name__,
        },
    )


def _restore_corrected(base: type, state: dict) -> Any:
    """Rebuild a corrected set; used by ``copy`` and ``pickle``."""
    obj = object.__new__(_corrected_class(base))
    obj.__dict__.update(state)
    return obj


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
        for name in ("species_column", "quality_column"):
            column = getattr(self, name)
            if column is not None and column not in meta.columns:
                raise KeyError(f"{name}={column!r} not found in MaldiSet metadata.")
        batch = meta[self.batch_column]
        species = meta[self.species_column] if self.species_column else None
        quality = meta[self.quality_column] if self.quality_column else None
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
            ``species`` / ``discrete_covariates`` and/or a ``quality=``
            argument). Species is not forwarded to Harmony's
            ``covariates``, which harmonypy removes like the batch.
        transformer_kwargs : dict, optional
            Extra keyword arguments forwarded to ``transformer_cls``.

        Returns
        -------
        MaldiSet
            Shallow-copied dataset with its ``_X_cache`` replaced by the
            corrected feature matrix. The ``spectra`` list is the same
            object in the returned ``MaldiSet``; labels / metadata are
            unchanged. Its ``filter`` returns subsets of the corrected
            matrix.

        Warns
        -----
        UserWarning
            If ``species_column`` is set but ``transformer_cls`` has no
            ``species`` / ``discrete_covariates`` argument and
            ``transformer_kwargs`` passes no covariate (``design``,
            ``covariates``, ...) either: species is then ignored. To protect
            species in Limma, pass a numeric ``design`` via
            ``transformer_kwargs``.
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

        if species is not None:
            protected = [
                n for n in ("species", "discrete_covariates") if n in init_params
            ]
            if protected:
                kwargs[protected[0]] = species
            elif not set(transformer_kwargs or {}) & _COVARIATE_KWARGS:
                warnings.warn(
                    f"{transformer_cls.__name__} has no protected-covariate "
                    f"argument (species / discrete_covariates); species_column "
                    f"is ignored.",
                    stacklevel=2,
                )
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
        new_ds._X_cache = X_corrected
        new_ds.meta = (
            ds.meta.loc[X_corrected.index].copy().rename_axis(ds.meta.index.name)
        )
        new_ds.__class__ = _corrected_class(
            getattr(type(ds), "_corrected_base", type(ds))
        )
        return new_ds
