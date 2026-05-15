"""Top-level package sanity tests."""

from __future__ import annotations

import maldibatchkit


def test_version_is_set():
    assert maldibatchkit.__version__


def test_author_is_set():
    assert maldibatchkit.__author__


def test_public_api():
    expected = {
        "BatchAwareWarping",
        "ComBat",
        "Harmony",
        "Limma",
        "MedianCentering",
        "QualityWeightedComBat",
        "ReferenceScaling",
        "SpeciesAwareComBat",
        "ZScorePerBatch",
    }
    assert expected.issubset(set(maldibatchkit.__all__))
    for name in expected:
        assert hasattr(maldibatchkit, name), name
