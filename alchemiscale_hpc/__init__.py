"""alchemiscale-hpc: Tools for using alchemiscale with HPC systems.

This package provides compute managers for autoscaling alchemiscale compute
services on various HPC queueing systems including SLURM, LSF, and PBS.

Backend implementations live in subpackages (``alchemiscale_hpc.slurm``,
``alchemiscale_hpc.lsf``, ``alchemiscale_hpc.pbs``) and self-register via
:func:`alchemiscale_hpc.base.register_backend` on import. The base interfaces
themselves are always available; backends are imported lazily.
"""

from .base import (
    HPCBatchApi,
    HPCManager,
    HPCManagerSettings,
    JobFailureError,
    JobNotFoundError,
    ScriptTemplateHPCManager,
    ScriptTemplateHPCManagerSettings,
    get_backend,
    list_backends,
    register_backend,
)

__all__ = [
    "HPCBatchApi",
    "HPCManager",
    "HPCManagerSettings",
    "JobFailureError",
    "JobNotFoundError",
    "ScriptTemplateHPCManager",
    "ScriptTemplateHPCManagerSettings",
    "get_backend",
    "list_backends",
    "register_backend",
]


def __getattr__(name: str):
    """Lazy attribute access for backend symbols.

    Accessing e.g. ``alchemiscale_hpc.SlurmManager`` imports the SLURM
    subpackage on demand without forcing every backend to be loaded at
    package import time.
    """
    _LAZY = {
        "SlurmManager": ("alchemiscale_hpc.slurm", "SlurmManager"),
        "SlurmManagerSettings": ("alchemiscale_hpc.slurm", "SlurmManagerSettings"),
        "SlurmBatchApi": ("alchemiscale_hpc.slurm", "SlurmBatchApi"),
    }
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module 'alchemiscale_hpc' has no attribute {name!r}")
