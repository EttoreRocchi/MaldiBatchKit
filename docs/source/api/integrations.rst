Integrations module
===================

Use MaldiBatchKit correctors directly on the data structures of
MaldiAMRKit.

MaldiSet adapter
----------------

.. autoclass:: maldibatchkit.integrations.MaldiSetAdapter
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from maldiamrkit import MaldiSet
   from maldibatchkit.integrations import MaldiSetAdapter
   from maldibatchkit import SpeciesAwareComBat

   # Load dataset (aggregated by an antibiotic label)
   ds = MaldiSet.from_directory(
       "spectra/", "metadata.csv",
       aggregate_by=dict(antibiotics="Ceftriaxone"),
   )

   # The adapter reads batch / species / quality from ds.meta
   adapter = MaldiSetAdapter(
       batch_column="Batch",
       species_column="Species",
       quality_column="SNR",
   )

   # Returns a NEW MaldiSet with corrected X and untouched labels
   corrected_ds = adapter.correct(ds, SpeciesAwareComBat)

   corrected_ds.X     # harmonised feature matrix
   corrected_ds.y     # AMR labels, unchanged

``batch_column``, ``species_column`` and ``quality_column`` name columns
of the metadata behind ``ds.meta``; a missing column raises ``KeyError``.
A corrector uses only what it accepts: ``quality_column`` has no effect
with :class:`~maldibatchkit.SpeciesAwareComBat`.
