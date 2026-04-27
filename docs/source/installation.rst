Installation
============

.. code-block:: bash

   pip install maldibatchkit

``maldiamrkit`` is a core dependency and is installed automatically -
``BatchAwareWarping`` reuses :class:`maldiamrkit.alignment.Warping`
under the hood, and :class:`~maldibatchkit.integrations.MaldiSetAdapter`
bridges to :class:`maldiamrkit.MaldiSet` for end-to-end AMR workflows.

Optional: Visualisation Extras
------------------------------

For UMAP before/after plots and the diagnostic summary figures:

.. code-block:: bash

   pip install maldibatchkit[viz]

This pulls in ``umap-learn`` and ``seaborn``. All plotting functions use
lazy imports, so ``matplotlib`` / ``umap`` are only required when a plot
is actually rendered.

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiBatchKit.git
   cd MaldiBatchKit
   pip install -e ".[dev,viz]"
   pre-commit install

See :doc:`contributing` for coding conventions, testing, and PR guidelines.

Documentation Build
-------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs,viz]"
   make -C docs html

The rendered site lands in ``docs/build/html/index.html``.
