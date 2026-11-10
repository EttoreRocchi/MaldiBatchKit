Corrections module
==================

Batch-effect correctors. Every class is a scikit-learn compatible
transformer with ``fit`` / ``transform`` / ``fit_transform`` - ``batch``
and covariates are passed at construction time and realigned to
``X.index`` on each call, so correctors are safe inside
``sklearn.pipeline.Pipeline`` and cross-validation without leakage.

Base class
----------

.. autoclass:: maldibatchkit.BaseBatchCorrector
   :members:
   :undoc-members:
   :show-inheritance:

All shipped correctors inherit from this class. Custom correctors
should follow the same pattern - see :doc:`/extending` for a full
walkthrough.

ComBat variants
---------------

The core ComBat implementation lives in
`combatlearn <https://github.com/EttoreRocchi/combatlearn>`_; MaldiBatchKit
re-exports the sklearn-compatible transformer and adds a thin
MALDI-specific preset for the species-as-covariate case.

.. autoclass:: maldibatchkit.ComBat
   :members:
   :undoc-members:
   :show-inheritance:

Species-aware ComBat
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maldibatchkit.SpeciesAwareComBat
   :members:
   :undoc-members:
   :show-inheritance:

Quality-weighted ComBat
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maldibatchkit.QualityWeightedComBat
   :members:
   :undoc-members:
   :show-inheritance:

Linear-model corrections
------------------------

.. autoclass:: maldibatchkit.Limma
   :members:
   :undoc-members:
   :show-inheritance:

Single-cell-style integration
-----------------------------

.. autoclass:: maldibatchkit.Harmony
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   Published Harmony has no test-time step. Here, ``transform`` applies a
   closed-form correction with the PCA, the cluster centroids and the
   per-batch correction frozen at ``fit``, and raises ``ValueError`` on a
   batch not seen at ``fit``.

Baselines
---------

Simple, auditable correctors useful as comparison points for more
sophisticated methods.

.. autoclass:: maldibatchkit.MedianCentering
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldibatchkit.ZScorePerBatch
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldibatchkit.ReferenceScaling
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldibatchkit.NoOpCorrector
   :members:
   :undoc-members:
   :show-inheritance:

Meta-corrector
--------------

:class:`~maldibatchkit.AutoCorrector` exposes ``method`` as a settable
hyperparameter so a single ``GridSearchCV`` can sweep across corrector
families and let the downstream classifier score pick the winner. See
:doc:`/choosing` for an end-to-end recipe.

.. autoclass:: maldibatchkit.AutoCorrector
   :members:
   :undoc-members:
   :show-inheritance:

MALDI-specific corrections
--------------------------

.. autoclass:: maldibatchkit.BatchAwareWarping
   :members:
   :undoc-members:
   :show-inheritance:

``BatchAwareWarping`` reuses :class:`maldiamrkit.alignment.Warping`
internally to warp each batch onto a shared global reference. Pair it
with an intensity-domain corrector (e.g. :class:`~maldibatchkit.ComBat`)
inside a :class:`sklearn.pipeline.Pipeline` for a full harmonisation:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from maldibatchkit import BatchAwareWarping, ComBat

   pipe = Pipeline([
       ("warp", BatchAwareWarping(batch=batch, method="piecewise")),
       ("combat", ComBat(batch=batch, method="fortin",
                         discrete_covariates=species)),
   ])
   X_corrected = pipe.fit_transform(X)
