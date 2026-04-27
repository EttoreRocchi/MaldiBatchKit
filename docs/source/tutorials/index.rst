Tutorials
=========

Interactive notebooks demonstrating MaldiBatchKit workflows end to end.
Each notebook is self-contained and can be downloaded from the
`GitHub repository <https://github.com/EttoreRocchi/MaldiBatchKit/tree/main/notebooks>`_.

.. toctree::
   :maxdepth: 2

   notebooks/01_quick_start
   notebooks/02_correction_methods
   notebooks/03_quality_weighted_combat
   notebooks/04_maldiset_integration
   notebooks/05_avoiding_data_leakage

Example Workflows
-----------------

Single-Call ComBat
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldibatchkit import ComBat
   from maldibatchkit.diagnostics import diagnostic_report

   corrector = ComBat(batch=batch, method="johnson")
   X_corrected = corrector.fit_transform(X)

   report = diagnostic_report(X, X_corrected, batch)
   print(report)

Train/Test Safe Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from maldibatchkit import SpeciesAwareComBat

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=batch, random_state=42,
   )

   pipe = Pipeline([
       ("combat", SpeciesAwareComBat(batch=batch, species=species)),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
   ])
   pipe.fit(X_train, y_train)
   score = pipe.score(X_test, y_test)

Chained Warping + ComBat
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from maldibatchkit import BatchAwareWarping, ComBat

   pipe = Pipeline([
       ("warp", BatchAwareWarping(batch=batch, method="piecewise",
                                  n_segments=8, max_shift=10)),
       ("combat", ComBat(batch=batch, method="fortin",
                         discrete_covariates=species)),
   ])
   X_corrected = pipe.fit_transform(X)

MaldiSet Adapter
~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldiamrkit import MaldiSet
   from maldibatchkit.integrations import MaldiSetAdapter
   from maldibatchkit import QualityWeightedComBat

   ds = MaldiSet.from_directory(
       "spectra/", "metadata.csv",
       aggregate_by=dict(antibiotics="Ceftriaxone"),
   )
   adapter = MaldiSetAdapter(
       batch_column="Batch",
       species_column="Species",
       quality_column="SNR",
   )
   corrected_ds = adapter.correct(ds, QualityWeightedComBat)
