"""Integrations with external MALDI toolkits.

Currently exposes :class:`MaldiSetAdapter`, a bridge to
:class:`maldiamrkit.MaldiSet`.
"""

from .maldiset import MaldiSetAdapter

__all__ = ["MaldiSetAdapter"]
