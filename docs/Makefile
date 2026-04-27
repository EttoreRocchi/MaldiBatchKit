# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
# Prefer the Python interpreter from the active virtualenv (if any) so
# `sphinx-build` resolves against the venv where the package itself is
# installed, rather than against a system-wide install on PATH that may
# be missing project dependencies.
PYTHON        ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python)
SPHINXBUILD   ?= $(PYTHON) -m sphinx
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
