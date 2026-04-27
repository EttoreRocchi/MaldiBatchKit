"""Import-error fallback paths for optional dependencies."""

from __future__ import annotations

import builtins
import sys

import pytest

from maldibatchkit.corrections import harmony as harmony_mod
from maldibatchkit.corrections import maldi as maldi_mod


def test_harmony_require_raises_when_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "harmonypy":
            raise ImportError("no harmonypy for you")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("harmonypy", None)
    with pytest.raises(ImportError, match="harmonypy"):
        harmony_mod.Harmony._require_harmonypy()


def test_require_maldiamrkit_raises_when_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "maldiamrkit":
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("maldiamrkit", None)
    with pytest.raises(ImportError, match="maldiamrkit"):
        maldi_mod._try_import_maldiamrkit()


def test_maldiset_adapter_raises_without_maldiamrkit(monkeypatch, tiny_dataset):
    from maldibatchkit.integrations import maldiset as adapter_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "maldiamrkit":
            raise ImportError("gone")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("maldiamrkit", None)
    with pytest.raises(ImportError, match="maldiamrkit"):
        adapter_mod._require_maldiset()
