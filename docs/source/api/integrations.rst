Integrations Module
===================

Bridges between MaldiBatchKit and sibling toolkits in the MaldiSuite
ecosystem.

MaldiSet Adapter
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
