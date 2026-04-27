"""Zenodo-hosted demo dataset loader for the MaldiBatchKit notebooks.

The notebooks in this folder share a single dataset: **MALDI-Kleb-AI**
(Rocchi et al. 2026, Zenodo DOI `10.5281/zenodo.17405072`), a 370 MB
archive of real MALDI-TOF mass spectra of *Klebsiella* isolates from
three Italian clinical centres, with Amikacin / Meropenem antimicrobial-
resistance annotations.

The helper in this module downloads the tarball once, caches it under
``~/.cache/maldibatchkit/`` (or the directory pointed to by the
``MALDIBATCHKIT_CACHE_DIR`` environment variable), extracts it, and
returns a :class:`DemoDataset` populated with:

* ``X``    -- binned feature matrix obtained via
  :class:`maldiamrkit.MaldiSet` (samples × m/z bins).
* ``meta`` -- per-sample metadata: ``Batch`` (= acquisition centre),
  ``Species`` (Klebsiella species), ``SNR`` (signal-to-noise ratio
  computed with :class:`maldiamrkit.preprocessing.quality.SpectrumQuality`),
  and the AMR labels ``Amikacin`` / ``Meropenem`` (``R``/``S``/``I``).
* ``mz``   -- m/z axis reported by ``MaldiSet`` (bin starts in Da).
* ``maldi_set`` -- the underlying ``MaldiSet`` object, so the
  ``MaldiSetAdapter`` notebook can reuse it directly.

This module lives exclusively under ``notebooks/`` and is intentionally
kept out of the installable ``maldibatchkit`` package.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from maldiamrkit import MaldiSet


__all__ = [
    "ZENODO_DOI",
    "ZENODO_RECORD_ID",
    "ZENODO_TAR_MD5",
    "ZENODO_TAR_URL",
    "DemoDataset",
    "get_cache_dir",
    "load_maldi_kleb_ai",
]


ZENODO_RECORD_ID = "17405072"
ZENODO_DOI = "10.5281/zenodo.17405072"
ZENODO_TAR_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/maldi-tof.tar?download=1"
)
ZENODO_TAR_MD5 = "c14b6c6b4210553962faa7f1dc27d275"

_DATASET_DIRNAME = "maldi-kleb-ai"
_TAR_NAME = "maldi-tof.tar"


def get_cache_dir() -> Path:
    """Resolve the root cache directory for MaldiBatchKit demo data.

    Priority:

    1. ``$MALDIBATCHKIT_CACHE_DIR`` environment variable (absolute path).
    2. ``~/.cache/maldibatchkit/`` (XDG-style default, cross-platform).

    The directory is created on first access.
    """
    env = os.environ.get("MALDIBATCHKIT_CACHE_DIR")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path.home() / ".cache" / "maldibatchkit"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_paths(cache_dir: Path | None = None) -> dict[str, Path]:
    root = (cache_dir or get_cache_dir()) / _DATASET_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "tar": root / _TAR_NAME,
        "extract": root / "extracted",
        "snr_cache": root / "snr.csv",
    }


def _md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dest: Path, *, verbose: bool = True) -> None:
    """Stream ``url`` to ``dest`` with an optional progress bar.

    We use ``urllib.request`` so the module keeps zero extra runtime
    dependencies.  A minimal progress line is printed unless
    ``verbose=False``.
    """
    tmp = dest.with_suffix(dest.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "maldibatchkit-demo"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        read = 0
        step = max(1 << 20, total // 50) if total else 1 << 20  # ~50 ticks
        last_mark = 0
        chunk_size = 1 << 16
        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            out.write(buf)
            read += len(buf)
            if verbose and total and read - last_mark >= step:
                pct = 100.0 * read / total
                print(f"  downloading maldi-tof.tar ... {pct:5.1f} %", end="\r")
                last_mark = read
    tmp.replace(dest)
    if verbose:
        print(" " * 60, end="\r")  # clear the progress line


def _ensure_tar(
    cache_dir: Path | None, *, force: bool = False, verbose: bool = True
) -> Path:
    paths = _dataset_paths(cache_dir)
    tar = paths["tar"]
    if force and tar.exists():
        tar.unlink()
    if tar.exists() and _md5_of(tar) == ZENODO_TAR_MD5:
        return tar
    if tar.exists():  # stale / corrupted
        tar.unlink()
    if verbose:
        print(
            f"Downloading MALDI-Kleb-AI from Zenodo (DOI {ZENODO_DOI}; "
            f"370 MB, one-shot) to {tar} ..."
        )
    _download_with_progress(ZENODO_TAR_URL, tar, verbose=verbose)
    got = _md5_of(tar)
    if got != ZENODO_TAR_MD5:
        tar.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {tar.name}: expected {ZENODO_TAR_MD5}, "
            f"got {got}. The download may be corrupted; re-run with "
            f"force_redownload=True."
        )
    return tar


def _ensure_extracted(
    cache_dir: Path | None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    paths = _dataset_paths(cache_dir)
    extract_dir = paths["extract"]
    sentinel = extract_dir / "metadata.csv"
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if sentinel.exists():
        return extract_dir
    tar = _ensure_tar(cache_dir, verbose=verbose)
    if verbose:
        print(f"Extracting {tar.name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, mode="r:*") as tf:
        tf.extractall(extract_dir, filter="data")
    if not sentinel.exists():
        raise RuntimeError(
            f"Extraction completed but {sentinel} is missing. "
            f"Is the tarball layout still 'spectra/*.txt + metadata.csv'?"
        )
    return extract_dir


@dataclass
class DemoDataset:
    """MALDI-Kleb-AI binned intensities + aligned metadata."""

    X: pd.DataFrame
    """Binned feature matrix, shape ``(n_samples, n_bins)``."""

    meta: pd.DataFrame
    """Per-sample metadata (``Batch``, ``Species``, ``SNR``,
    ``Amikacin``, ``Meropenem``), indexed by ``X.index``."""

    mz: np.ndarray
    """m/z axis as reported by ``MaldiSet`` (bin-start values in Da,
    e.g. ``[2000, 2003, ..., 19997]`` for the default 2000-20000 Da
    window with ``bin_width=3``). Length ``n_bins``."""

    info: dict = field(default_factory=dict)
    """Provenance info: Zenodo DOI, MD5 checksum, loader arguments."""

    maldi_set: Any = None
    """Underlying ``maldiamrkit.MaldiSet`` for notebook 4."""

    @property
    def batch(self) -> pd.Series:
        """Alias for ``meta['Batch']`` (acquisition centre)."""
        return self.meta["Batch"]

    @property
    def species(self) -> pd.Series:
        """Alias for ``meta['Species']``."""
        return self.meta["Species"]

    @property
    def quality(self) -> pd.Series:
        """Alias for ``meta['SNR']`` (signal-to-noise ratio)."""
        return self.meta["SNR"]


def _compute_snr_table(
    ds: MaldiSet,
    cache_path: Path,
    *,
    verbose: bool,
) -> pd.Series:
    """Per-spectrum SNR via maldiamrkit's SpectrumQuality, cached on disk."""
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0).iloc[:, 0]

    from maldiamrkit.preprocessing.quality import SpectrumQuality

    sq = SpectrumQuality()
    if verbose:
        print(f"Computing SNR for {len(ds.spectra)} spectra (one-shot)...")
    rows: dict[str, float] = {}
    for spec in ds.spectra:
        try:
            rows[spec.id] = float(sq.assess(spec).snr)
        except Exception:  # pragma: no cover - stray corrupt spectrum
            rows[spec.id] = float("nan")
    ser = pd.Series(rows, name="SNR")
    ser.to_frame().to_csv(cache_path)
    return ser


def load_maldi_kleb_ai(
    *,
    antibiotic: str = "Amikacin",
    bin_width: int = 3,
    cache_dir: Path | None = None,
    force_redownload: bool = False,
    compute_snr: bool = True,
    verbose: bool = False,
) -> DemoDataset:
    """Download (once) and return the MALDI-Kleb-AI demo dataset.

    Parameters
    ----------
    antibiotic : {'Amikacin', 'Meropenem'}, default='Amikacin'
        Which AMR column to expose as the primary label in
        ``ds.maldi_set.y``.  Both columns are kept in ``ds.meta``
        regardless.
    bin_width : int, default=3
        Bin width in Daltons forwarded to
        :meth:`maldiamrkit.MaldiSet.from_directory`.
    cache_dir : Path, optional
        Root cache directory override.  When ``None``, the module uses
        ``$MALDIBATCHKIT_CACHE_DIR`` if set, otherwise
        ``~/.cache/maldibatchkit/``.
    force_redownload : bool, default=False
        Re-download the tarball and re-extract even if a valid cache
        already exists.  Useful for troubleshooting or after a Zenodo
        record update.
    compute_snr : bool, default=True
        If ``True``, populate ``meta['SNR']`` with per-sample
        signal-to-noise ratios computed by
        :class:`maldiamrkit.preprocessing.quality.SpectrumQuality`.
        Disable for a faster load if you do not need
        :class:`maldibatchkit.QualityWeightedComBat`.
    verbose : bool, default=False
        Print progress for the download, extraction, and SNR steps.

    Returns
    -------
    DemoDataset
        With ``X``, ``meta`` (Batch / Species / SNR / Amikacin /
        Meropenem), ``mz``, and the underlying ``maldi_set``.

    Notes
    -----
    * The dataset is **real clinical data** (Klebsiella isolates from
      Rome, Milan, Catania); please cite the Zenodo record if you reuse
      it. DOI: ``10.5281/zenodo.17405072``.
    * First call downloads 370 MB; subsequent calls are millisecond-fast
      (the loader re-uses the extracted spectra + a cached SNR CSV).
    """
    try:
        from maldiamrkit import MaldiSet
    except ImportError as exc:  # pragma: no cover - maldiamrkit is a core dep
        raise ImportError(
            "load_maldi_kleb_ai requires maldiamrkit. Reinstall with "
            "`pip install -U maldibatchkit`."
        ) from exc

    if antibiotic not in ("Amikacin", "Meropenem"):
        raise ValueError(
            f"antibiotic must be one of 'Amikacin' / 'Meropenem', got {antibiotic!r}."
        )

    paths = _dataset_paths(cache_dir)
    extract_dir = _ensure_extracted(
        cache_dir, force=force_redownload, verbose=verbose
    )
    metadata_csv = extract_dir / "metadata.csv"
    spectra_dir = extract_dir / "spectra"

    # Build a MaldiSet from the extracted directory -- MaldiSet handles
    # metadata alignment, ID mismatches, and binning in one call.
    ds = MaldiSet.from_directory(
        str(spectra_dir),
        str(metadata_csv),
        aggregate_by={"antibiotics": [antibiotic]},
        bin_width=bin_width,
        verbose=verbose,
    )
    X = ds.X
    meta = ds.meta.loc[X.index].copy()

    # Normalise metadata columns to the canonical names used by the
    # other notebooks:  City -> Batch.  Species / Amikacin / Meropenem
    # stay as-is; they already match.
    if "City" in meta.columns:
        meta = meta.rename(columns={"City": "Batch"})

    # Optional SNR column (cached so we only compute it once per tarball).
    if compute_snr:
        snr_full = _compute_snr_table(ds, paths["snr_cache"], verbose=verbose)
        meta["SNR"] = snr_full.reindex(meta.index).astype(float)
    else:
        meta["SNR"] = float("nan")

    # The canonical m/z axis is the one ``MaldiSet`` uses to label the
    # feature matrix's columns (bin starts produced by the default
    # preprocessing pipeline -- e.g. 2000, 2003, ..., 19997 for the
    # 2000-20000 Da / 3-Da window).  We read it directly from ``X``.
    mz = np.asarray(X.columns, dtype=float)

    info = {
        "source": "Zenodo MALDI-Kleb-AI",
        "doi": ZENODO_DOI,
        "record_id": ZENODO_RECORD_ID,
        "md5_tar": ZENODO_TAR_MD5,
        "n_samples": X.shape[0],
        "n_bins": X.shape[1],
        "bin_width": bin_width,
        "antibiotic": antibiotic,
        "cache_dir": str(paths["root"]),
    }
    return DemoDataset(
        X=X, meta=meta, mz=mz, info=info, maldi_set=ds
    )
