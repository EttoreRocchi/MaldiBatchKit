Installation
============

.. code-block:: bash

   pip install maldibatchkit

MaldiBatchKit requires Python 3.10 or newer.

``maldiamrkit`` is a core dependency and is installed automatically -
``BatchAwareWarping`` reuses :class:`maldiamrkit.alignment.Warping`
under the hood, and :class:`~maldibatchkit.integrations.MaldiSetAdapter`
applies any corrector to a :class:`maldiamrkit.MaldiSet`.

Install the full MaldiSuite
---------------------------

To install MaldiBatchKit together with
`MaldiAMRKit <https://github.com/EttoreRocchi/MaldiAMRKit>`__ and
`MaldiDeepKit <https://github.com/EttoreRocchi/MaldiDeepKit>`__ at
compatible versions, install the
`maldisuite <https://pypi.org/project/maldisuite/>`__ meta-package:

.. code-block:: bash

   pip install maldisuite

Optional: visualisation extras
------------------------------

For UMAP before/after plots and the diagnostic summary figures:

.. code-block:: bash

   pip install maldibatchkit[viz]

This pulls in ``umap-learn`` and ``seaborn``. ``umap-learn`` is imported
only when :func:`~maldibatchkit.viz.plot_batch_umap` runs; ``matplotlib``
is a core dependency.

Development installation
------------------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiBatchKit.git
   cd MaldiBatchKit
   pip install -e ".[dev,viz]"
   pre-commit install

See :doc:`contributing` for coding conventions, testing, and PR guidelines.

Documentation build
-------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs,viz]"
   make -C docs html

The rendered site lands in ``docs/build/html/index.html``.
