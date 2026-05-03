"""SLURM-specific compute manager for alchemiscale."""

from .manager import SlurmManager, SlurmBatchApi
from .settings import SlurmManagerSettings

__all__ = ["SlurmManager", "SlurmBatchApi", "SlurmManagerSettings"]
