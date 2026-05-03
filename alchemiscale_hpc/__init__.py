"""
alchemiscale-hpc: Tools for using alchemiscale with HPC systems.

This package provides compute managers for autoscaling alchemiscale compute
services on various HPC queueing systems including SLURM, LSF, and PBS.
"""

from .base import (
    HPCManager,
    HPCManagerSettings,
    HPCBatchApi,
    JobNotFoundError,
    JobFailureError,
)

# Import system-specific implementations
from .slurm import SlurmManager, SlurmManagerSettings, SlurmBatchApi

__all__ = [
    # Base classes
    "HPCManager",
    "HPCManagerSettings",
    "HPCBatchApi",
    "JobNotFoundError",
    "JobFailureError",
    # SLURM implementation
    "SlurmManager",
    "SlurmManagerSettings",
    "SlurmBatchApi",
]
