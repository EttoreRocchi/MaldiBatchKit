"""Shared test fixtures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _make_dataset(
    rng: np.random.Generator,
    *,
    n_per_batch: int = 30,
    n_batches: int = 2,
    n_features: int = 20,
    batch_shift: float = 1.0,
    species_shift: float = 0.3,
) -> dict[str, object]:
    """Build a small synthetic MALDI-like feature matrix with batch + species."""
    batch_labels: list[object] = []
    species_labels: list[object] = []
    offsets = []
    species_offsets = []
    for i in range(n_batches):
        batch_labels.extend([f"b{i}"] * n_per_batch)
        species_batch = np.repeat(
            [f"sp{j}" for j in range(2)], n_per_batch // 2
        ).tolist()
        species_labels.extend(species_batch)
        offsets.extend([batch_shift * (i - (n_batches - 1) / 2)] * n_per_batch)
        species_offsets.extend(
            [species_shift if s == "sp1" else -species_shift for s in species_batch]
        )
    n = n_per_batch * n_batches
    X = rng.standard_normal((n, n_features))
    X = X + np.asarray(offsets)[:, None] + np.asarray(species_offsets)[:, None]
    idx = [f"s{i:04d}" for i in range(n)]
    return {
        "X": pd.DataFrame(X, index=idx),
        "batch": pd.Series(batch_labels, index=idx, name="batch"),
        "species": pd.Series(species_labels, index=idx, name="species"),
    }


@pytest.fixture
def tiny_dataset(rng):
    return _make_dataset(rng)


@pytest.fixture
def big_dataset(rng):
    return _make_dataset(
        rng, n_per_batch=40, n_batches=3, n_features=50, batch_shift=1.5
    )
