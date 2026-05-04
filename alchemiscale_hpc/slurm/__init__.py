"""SLURM-specific compute manager for alchemiscale."""

from ..base import register_backend
from .manager import SlurmBatchApi, SlurmManager
from .settings import SlurmManagerSettings

# Register this backend so the CLI and lookup helpers can find it.
register_backend("slurm", SlurmManager, SlurmManagerSettings, SlurmBatchApi)

__all__ = ["SlurmManager", "SlurmBatchApi", "SlurmManagerSettings"]
