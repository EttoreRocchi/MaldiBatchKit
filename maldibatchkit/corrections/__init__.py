"""Batch-effect correction transformers (all train/test safe)."""

from .auto import METHOD_REGISTRY, AutoCorrector
from .baselines import MedianCentering, ReferenceScaling, ZScorePerBatch
from .combat import ComBat, SpeciesAwareComBat
from .harmony import Harmony
from .limma import Limma
from .maldi import BatchAwareWarping
from .noop import NoOpCorrector
from .quality_weighted import QualityWeightedComBat

__all__ = [
    "METHOD_REGISTRY",
    "AutoCorrector",
    "BatchAwareWarping",
    "ComBat",
    "Harmony",
    "Limma",
    "MedianCentering",
    "NoOpCorrector",
    "QualityWeightedComBat",
    "ReferenceScaling",
    "SpeciesAwareComBat",
    "ZScorePerBatch",
]
