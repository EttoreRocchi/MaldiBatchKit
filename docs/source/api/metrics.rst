Metrics Module
==============

Batch-aware downstream classifier metrics. Each function computes a
sklearn metric **per batch** on the held-out fold, then aggregates the
per-batch scalars into a single number according to a chosen weighting
scheme. They are the right way to score a corrector when the goal is a
model that performs well *on every site*, not one that just maximises
the pooled metric.

.. contents::
   :local:

Weighting modes
---------------

Every ``batch_*_score`` function accepts ``weights=``:

* ``'uniform'`` - each batch contributes one entry with equal weight
  (``w_i = 1 / n_batches``), regardless of its sample count. The right
  default when the goal is a corrector that generalises across sites.
* ``'balanced'`` - weights inversely proportional to per-batch size
  (``w_i ∝ 1 / n_i``). The **smallest site gets the loudest voice**,
  mirroring sklearn's ``class_weight='balanced'`` formula applied at
  the per-batch level. Use when minority sites are the hardest
  distributions to learn and the corrector must protect them.
* ``'size'`` - per-batch weights proportional to sample counts.
  The relationship to the standard pooled value depends on the
  metric family:

  - **Additive** ``correct / total`` **metrics** (accuracy, error
    rate): ``'size'`` exactly recovers the pooled value.
  - **Class-conditional rates** (sensitivity, specificity): recovering
    the pooled value needs *class-conditional* weights (positives-
    per-batch for sensitivity, negatives-per-batch for specificity),
    not total-sample weights. Equal only when class prevalence is
    constant across batches.
  - **Non-linear metrics** (F1, precision, balanced accuracy, MCC):
    ratios of sums or non-linear functions of confusion-matrix counts.
    No per-batch weighting recovers the pooled value exactly.
  - **Pairwise / rank-based metrics** (AUROC, average precision):
    pooled values count *cross-batch* positive-vs-negative pairs,
    which any per-batch reducer **cannot see by construction** -
    those pairs are gone, not approximated.

  In every non-additive case ``'size'`` still gives the dominant
  site the loudest voice - useful when you want to score the
  classifier as a single-site practitioner would.
* mapping ``{batch_label: weight}`` or array aligned with
  ``np.unique(batch)`` - explicit per-batch weights for any custom
  policy.

Within each batch, the wrapped sklearn metric's own averaging parameter
(``average='binary'``, ``'macro'``, ``'weighted'``, ``'micro'``) is
preserved untouched - ``weights=`` only controls the *across-batch*
reducer.

Degenerate folds (a batch containing only one class for AUROC / average
precision) are **silently dropped from the aggregate with a warning**
and the remaining weights renormalised.

Per-batch metric functions
--------------------------

.. autofunction:: maldibatchkit.metrics.batch_roc_auc_score

.. autofunction:: maldibatchkit.metrics.batch_average_precision_score

.. autofunction:: maldibatchkit.metrics.batch_balanced_accuracy_score

.. autofunction:: maldibatchkit.metrics.batch_matthews_corrcoef

.. autofunction:: maldibatchkit.metrics.batch_f1_score

.. autofunction:: maldibatchkit.metrics.batch_precision_score

.. autofunction:: maldibatchkit.metrics.batch_recall_score

Scorer factory
--------------

:func:`~maldibatchkit.metrics.make_batch_scorer` wraps a per-batch
metric and an aggregation into a sklearn-compatible
``scorer(estimator, X, y)`` callable suitable for
``GridSearchCV(scoring=...)``. It captures the full ``batch`` Series at
factory time and slices it by ``y.index`` for every fold; pass ``y``
as a :class:`pandas.Series` indexed by the sample IDs used by ``X`` and
``batch`` for safe alignment.

.. autofunction:: maldibatchkit.metrics.make_batch_scorer

Example
-------

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import GridSearchCV, StratifiedKFold
   from sklearn.pipeline import Pipeline

   from maldibatchkit import AutoCorrector
   from maldibatchkit.metrics import make_batch_scorer

   # Per-batch AUROC, each site weighted equally.
   scorer = make_batch_scorer(batch, metric="roc_auc", weights="uniform")

   pipe = Pipeline([
       ("correct", AutoCorrector(batch=batch,
                                 discrete_covariates=species)),
       ("clf", LogisticRegression(max_iter=1000)),
   ])

   grid = GridSearchCV(
       pipe,
       param_grid={"correct__method": ["noop", "combat-fortin", "harmony"]},
       scoring=scorer,
       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
   )
   grid.fit(X, y)
   print(grid.best_params_, grid.best_score_)

Comparing ``weights='size'`` (pooled, dominant-site biased) and
``weights='uniform'`` (every site counts equally) frequently picks
**different** correctors. Use the comparison to make the trade-off
explicit before deploying a corrector.
