Extending MaldiBatchKit
=======================

Every corrector shipped with MaldiBatchKit is a subclass of
:class:`~maldibatchkit.BaseBatchCorrector`. Custom correctors plug into
the same contract: subclass the base, implement two methods, and you
get a scikit-learn compatible, train/test-safe transformer for free.

Why Inherit?
------------

:class:`~maldibatchkit.BaseBatchCorrector` solves the two awkward parts
of sklearn + batch correction:

1. **Index alignment.** ``batch`` (and any covariates) is passed once at
   construction time. On every ``fit`` / ``transform`` call the stored
   series is aligned to ``X.index``, so sklearn pipelines and
   cross-validation can pass only ``X`` around - you never have to
   manually slice ``batch`` for train / test folds.
2. **Leakage safety.** ``fit`` learns parameters only from the rows it
   actually receives; ``transform`` never re-fits. Fitted attributes
   (those ending in ``_``) are used verbatim on unseen rows, even when
   a new batch level appears in the test set.

The base class additionally handles NaN / finite checks,
``DataFrame``-vs-``ndarray`` round-tripping, and populates
``feature_names_in_`` / ``n_features_in_`` / ``get_feature_names_out``
so your corrector behaves like any other sklearn transformer.

Minimal Subclass
----------------

Two methods are required: :meth:`~maldibatchkit.BaseBatchCorrector._fit_impl`
to learn parameters, and
:meth:`~maldibatchkit.BaseBatchCorrector._transform_impl` to apply them.
Store fitted attributes as ``..._`` members so
:func:`sklearn.utils.validation.check_is_fitted` picks them up:

.. code-block:: python

   import pandas as pd
   from maldibatchkit import BaseBatchCorrector

   class MeanCentering(BaseBatchCorrector):
       """Subtract per-batch means from each feature."""

       def _fit_impl(self, X_df, batch):
           self.batch_means_ = X_df.groupby(batch).mean()
           self.grand_mean_ = X_df.mean(axis=0)

       def _transform_impl(self, X_df, batch):
           out = X_df.copy().astype(float)
           known = set(self.batch_means_.index)
           for lvl in pd.unique(batch):
               mask = batch == lvl
               offset = (
                   self.batch_means_.loc[lvl].to_numpy()
                   if lvl in known
                   else self.grand_mean_.to_numpy()
               )
               out.loc[mask] = out.loc[mask].to_numpy() - offset
           return out

The inherited :meth:`~maldibatchkit.BaseBatchCorrector.fit`,
:meth:`~maldibatchkit.BaseBatchCorrector.transform`, and
``fit_transform`` (via ``TransformerMixin``) do the rest:

.. code-block:: python

   corrector = MeanCentering(batch=batch)
   X_corrected = corrector.fit_transform(X)

Adding Hyperparameters
----------------------

Override ``__init__`` to add method-specific hyperparameters. Always
call ``super().__init__(batch=batch)`` - the base class stores ``batch``
under ``self.batch`` so
:meth:`~sklearn.base.BaseEstimator.get_params` and
:meth:`~sklearn.base.BaseEstimator.set_params` work:

.. code-block:: python

   from maldibatchkit._utils import ArrayLike

   class ScaledMeanCentering(BaseBatchCorrector):
       """Mean centering with a user-controlled scale factor."""

       def __init__(self, batch: ArrayLike, *, alpha: float = 1.0) -> None:
           super().__init__(batch=batch)
           self.alpha = alpha

       def _fit_impl(self, X_df, batch):
           self.batch_means_ = X_df.groupby(batch).mean()

       def _transform_impl(self, X_df, batch):
           out = X_df.copy().astype(float)
           for lvl in pd.unique(batch):
               mask = batch == lvl
               offset = self.batch_means_.loc[lvl].to_numpy() * self.alpha
               out.loc[mask] = out.loc[mask].to_numpy() - offset
           return out

Dropping It Into a Pipeline
---------------------------

Custom correctors are drop-in sklearn ``Pipeline`` steps. Because the
base class enforces train/test separation, you can safely
cross-validate without leakage:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score

   pipe = Pipeline([
       ("mean", MeanCentering(batch=batch)),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier(n_estimators=200)),
   ])
   scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")

Adding Covariates
-----------------

If your corrector needs more than ``batch`` (for example species or a
quality score), follow the same pattern the shipped correctors use:
take the auxiliary vector in ``__init__``, store it on ``self``, and
subset it inside ``_fit_impl`` / ``_transform_impl`` via
:func:`maldibatchkit._utils._subset`:

.. code-block:: python

   from maldibatchkit._utils import _subset

   class QualityWeightedMean(BaseBatchCorrector):
       def __init__(self, batch, *, quality) -> None:
           super().__init__(batch=batch)
           self.quality = quality

       def _fit_impl(self, X_df, batch):
           q = _subset(self.quality, X_df.index).to_numpy(dtype=float)
           # weighted per-batch mean -- low quality contributes less
           self.means_ = (
               (X_df.mul(q, axis=0)).groupby(batch).sum()
               .div(pd.Series(q).groupby(batch).sum(), axis=0)
           )

       def _transform_impl(self, X_df, batch):
           out = X_df.copy().astype(float)
           for lvl in pd.unique(batch):
               mask = batch == lvl
               out.loc[mask] = (
                   out.loc[mask].to_numpy() - self.means_.loc[lvl].to_numpy()
               )
           return out

See :class:`~maldibatchkit.QualityWeightedComBat` for a production-grade
example of the same pattern (iterative empirical-Bayes with quality
weights).

Conventions
-----------

See :doc:`contributing` for the full style guide. The important points
for custom correctors:

- **NumPy-style docstrings** on every public class (rendered by
  napoleon).
- **Trailing-underscore fitted attributes** (``self.batch_means_``, not
  ``self.means``) - required by sklearn's
  :func:`~sklearn.utils.validation.check_is_fitted`.
- **No side effects outside ``fit``.** ``transform`` must be idempotent:
  calling it twice on the same input must return the same output.
- **Unseen batches** at transform time should fall back to a training
  statistic (grand mean, identity, ...) rather than raising, so
  cross-validation folds that split off a full batch still work.
- **Clear ``ImportError`` for optional dependencies.** If your
  corrector wraps a library that is not a hard dependency, raise an
  explanatory ``ImportError`` rather than letting a bare
  ``ModuleNotFoundError`` propagate. See
  :class:`~maldibatchkit.Harmony` for the reference pattern.

Reference Implementations
-------------------------

The simplest real-world examples are in the ``baselines`` module - read
them top-to-bottom to see the idiom in a few dozen lines:

- :class:`~maldibatchkit.MedianCentering` - fitted table + grand-median
  fallback.
- :class:`~maldibatchkit.ZScorePerBatch` - per-batch mean and stddev
  plus an ``eps`` hyperparameter.
- :class:`~maldibatchkit.ReferenceScaling` - non-trivial reference
  selection logic at ``fit`` time.

For a corrector with an iterative ``fit`` and an auxiliary covariate,
study :class:`~maldibatchkit.QualityWeightedComBat`.
