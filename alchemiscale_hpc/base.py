"""Base classes and interfaces for HPC batch system managers.

This module provides abstract base classes that define the interface
for HPC-specific compute managers. Each queueing system (SLURM, LSF, PBS, etc.)
should implement these interfaces.

Two abstraction layers are provided:

* :class:`HPCManager` --- the bare-minimum manager contract. Subclasses must
  implement :meth:`HPCManager.create_compute_services` (inherited as abstract
  from :class:`alchemiscale.compute.manager.ComputeManager`) and
  :meth:`HPCManager._create_batch_api`.
* :class:`ScriptTemplateHPCManager` --- a concrete subclass that implements
  the common "render a job script from a template, then submit it" workflow
  used by all current backends (SLURM, LSF, PBS). Subclasses only need to
  implement :meth:`ScriptTemplateHPCManager._create_job_script`.

A backend implementation typically extends :class:`ScriptTemplateHPCManager`
together with :class:`HPCBatchApi` and :class:`ScriptTemplateHPCManagerSettings`.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set
from uuid import uuid4

import yaml
from pydantic import Field

from alchemiscale.compute.manager import ComputeManager
from alchemiscale.compute.settings import ComputeManagerSettings, ComputeServiceSettings

# Registry of available backends, populated by backend modules at import time.
# Maps backend name (e.g. "slurm") -> (Manager class, Settings class, BatchApi class).
_BACKENDS: Dict[str, tuple] = {}


def register_backend(name: str, manager_cls, settings_cls, batch_api_cls) -> None:
    """Register an HPC backend so the CLI and lookup helpers can find it.

    Each backend module should call this at import time so that
    ``alchemiscale-hpc <name> ...`` commands work without modifications to
    :mod:`alchemiscale_hpc.cli`.

    Parameters
    ----------
    name
        Short identifier for the backend (e.g. ``"slurm"``, ``"lsf"``, ``"pbs"``).
    manager_cls
        Concrete :class:`HPCManager` subclass.
    settings_cls
        Concrete :class:`HPCManagerSettings` subclass.
    batch_api_cls
        Concrete :class:`HPCBatchApi` subclass.
    """
    _BACKENDS[name] = (manager_cls, settings_cls, batch_api_cls)


def get_backend(name: str) -> tuple:
    """Return the registered ``(Manager, Settings, BatchApi)`` triple for ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not a registered backend.
    """
    if name not in _BACKENDS:
        raise KeyError(
            f"Backend {name!r} is not registered. Known backends: "
            f"{sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]


def list_backends() -> List[str]:
    """Return the names of all registered backends."""
    return sorted(_BACKENDS)


class JobNotFoundError(Exception):
    """Raised when a job is not found in the batch queue."""


class JobFailureError(Exception):
    """Raised when a batch job has failed."""


class HPCManagerSettings(ComputeManagerSettings):
    """Base settings class for HPC compute managers.

    Extends :class:`alchemiscale.compute.settings.ComputeManagerSettings`
    with the minimum HPC-specific configuration that every batch backend
    needs: a job-name prefix and a per-cycle submission cap.
    """

    job_name_prefix: str = Field(
        "alchemiscale",
        description=("Prefix for batch job names. Full name will be <prefix>-<uuid>."),
    )
    max_submit_per_cycle: int = Field(
        1,
        description="Maximum number of jobs to submit in a single manager cycle.",
    )


class ScriptTemplateHPCManagerSettings(HPCManagerSettings):
    """Settings for HPC managers that submit jobs by rendering a script template.

    All current backends (SLURM, LSF, PBS) use this pattern, so a backend
    typically subclasses this rather than :class:`HPCManagerSettings` directly.
    """

    job_script_template: Path = Field(
        ...,
        description="Path to batch job script template file.",
    )
    job_script_dir: Path | None = Field(
        None,
        description=(
            "Directory in which to write rendered job scripts. If ``None``, "
            "the system temp directory is used. Rendered scripts are removed "
            "after submission unless ``keep_job_scripts`` is set."
        ),
    )
    keep_job_scripts: bool = Field(
        False,
        description=(
            "If True, do not delete rendered job scripts after submission. "
            "Useful for debugging. Default False."
        ),
    )
    cleanup_successful_jobs: bool = Field(
        True,
        description=(
            "If True, the per-cycle ``clear_successful_jobs`` step is enabled. "
            "Backends interpret this differently: the SLURM backend drops "
            "completed jobs from its in-memory tracking set; the (future) "
            "Kubernetes-style backends actually delete completed Job objects."
        ),
    )
    cleanup_failed_jobs: bool = Field(
        True,
        description=(
            "If True, ``clear_failed_jobs`` is enabled. Same backend-specific "
            "semantics as ``cleanup_successful_jobs``."
        ),
    )


class HPCBatchApi(ABC):
    """Abstract base class for HPC batch system APIs.

    Each queueing system (SLURM, LSF, PBS) should implement this interface
    to provide a consistent way to interact with the batch system.
    """

    def __init__(self, settings: HPCManagerSettings):
        """Initialize HPC batch API.

        Parameters
        ----------
        settings
            Settings for the HPC manager.
        """
        self.settings = settings
        self.tracked_jobs: Set[str] = set()

    @abstractmethod
    def check_job_health(self) -> None:
        """Raise :class:`JobFailureError` if any tracked job has failed."""

    @abstractmethod
    def verify_running_jobs(self, server_job_names: Set[str]) -> None:
        """Verify that all running jobs are registered with the server.

        Parameters
        ----------
        server_job_names
            Set of job names reported by the alchemiscale server.

        Raises
        ------
        JobNotFoundError
            If a running job is not registered with the server.
        """

    @abstractmethod
    def clear_successful_jobs(self) -> None:
        """Per-cycle cleanup hook for jobs that completed successfully.

        The exact action is backend-defined. For SLURM-style backends this is
        an in-memory operation (drop the IDs from ``tracked_jobs``); for
        Kubernetes-style backends this should actually delete the Job
        objects from the batch system. Either way, this is called
        unconditionally in :meth:`ScriptTemplateHPCManager.create_compute_services`
        once per cycle, gated only by the ``cleanup_successful_jobs`` setting.
        """

    @abstractmethod
    def clear_failed_jobs(self) -> None:
        """Operator-driven cleanup hook for failed jobs.

        Mirrors :meth:`clear_successful_jobs` but for failed jobs. Not called
        on every cycle; reached via the ``alchemiscale-hpc <backend> cleanup
        --failed`` CLI command after an operator has investigated a failure.
        """

    @abstractmethod
    def jobs_pending(self) -> bool:
        """Return True if any tracked jobs are in a pending (not-yet-running) state."""

    @abstractmethod
    def get_jobs(self) -> List[Dict[str, str]]:
        """Get all jobs in the queue.

        Returns
        -------
        list of dict
            Each dict contains at minimum ``job_id``, ``name``, ``state``.
        """

    @abstractmethod
    def submit_job(self, script_path: Path) -> str:
        """Submit ``script_path`` to the batch system and return the job ID."""


class HPCManager(ComputeManager, ABC):
    """Abstract base class for HPC-specific compute managers.

    Subclasses must implement :meth:`_create_batch_api` and
    :meth:`create_compute_services` (inherited from
    :class:`alchemiscale.compute.manager.ComputeManager`).

    For backends that follow the standard "render script template, submit
    via batch system" pattern, prefer subclassing
    :class:`ScriptTemplateHPCManager` instead.
    """

    def __init__(self, settings: HPCManagerSettings, service_settings_path: Path):
        """Initialize an HPC manager.

        Parameters
        ----------
        settings
            Settings for the HPC manager.
        service_settings_path
            Path to YAML file containing
            :class:`alchemiscale.compute.settings.ComputeServiceSettings` for
            services launched by this manager.
        """
        with open(service_settings_path, "r") as f:
            service_settings_dict = yaml.safe_load(f)

        # ComputeManager expects a ComputeServiceSettings instance, not a raw
        # dict (it sets `compute_manager_id` as an attribute on it).
        service_settings = ComputeServiceSettings(**service_settings_dict)
        super().__init__(settings=settings, service_settings=service_settings)

        # Subclasses provide the concrete batch API via _create_batch_api().
        self.batch_api: HPCBatchApi = self._create_batch_api()

    @abstractmethod
    def _create_batch_api(self) -> HPCBatchApi:
        """Construct and return the backend-specific :class:`HPCBatchApi`."""


class ScriptTemplateHPCManager(HPCManager):
    """HPC manager that submits jobs by rendering a job-script template.

    Implements the common ``create_compute_services`` workflow shared by the
    SLURM/LSF/PBS backends:

    1. Health-check tracked jobs.
    2. Verify running jobs are registered with the server.
    3. Clear successful jobs (semantics are backend-specific; see
       :meth:`HPCBatchApi.clear_successful_jobs`).
    4. If no jobs are pending, submit up to ``max_submit_per_cycle`` new jobs.

    Subclasses only need to implement :meth:`_create_job_script`, which renders
    the template into a script ready for ``submit_job``.
    """

    settings: ScriptTemplateHPCManagerSettings

    def __init__(
        self,
        settings: ScriptTemplateHPCManagerSettings,
        service_settings_path: Path,
    ):
        super().__init__(settings=settings, service_settings_path=service_settings_path)

        with open(self.settings.job_script_template, "r") as f:
            self.job_script_template: str = f.read()

    def create_compute_services(self, data: dict) -> int:
        """Create compute services by submitting batch jobs.

        Called by the parent :class:`ComputeManager` when scaling up is needed.

        Parameters
        ----------
        data
            Data from the server instruction containing:

            * ``compute_service_ids``: list of currently active service IDs
            * ``num_tasks``: number of waiting tasks

        Returns
        -------
        int
            Number of new compute services created.
        """
        compute_service_ids = data["compute_service_ids"]
        # Each ComputeServiceID has form "{name}-{uuid_hex}". Strip only the
        # trailing hex suffix so that service `name` (which may itself contain
        # hyphens, e.g. "alchemiscale-<uuid4>") is preserved intact.
        server_job_names = {csid.rsplit("-", 1)[0] for csid in compute_service_ids}

        self.logger.info("Checking health of batch jobs")
        self.batch_api.check_job_health()

        self.logger.info("Verifying running jobs are registered with server")
        self.batch_api.verify_running_jobs(server_job_names)

        self.logger.info("Clearing successful jobs")
        self.batch_api.clear_successful_jobs()

        if self.batch_api.jobs_pending():
            self.logger.info("Skipping job creation, pending jobs exist")
            return 0

        num_submitted = 0
        for _ in range(self.settings.max_submit_per_cycle):
            job_script_path = self._create_job_script()
            try:
                job_id = self.batch_api.submit_job(job_script_path)
                self.logger.info(f"Submitted batch job {job_id}")
                num_submitted += 1
            except Exception as e:
                self.logger.error(f"Failed to submit job: {e}")
                break
            finally:
                if not self.settings.keep_job_scripts:
                    try:
                        os.unlink(job_script_path)
                    except OSError:
                        pass
        return num_submitted

    def _generate_job_name(self) -> str:
        """Return a unique job name of the form ``{prefix}-{uuid4}``.

        The UUID portion uses the dashed string form, so the full name is
        ``<prefix>-<8-4-4-4-12 hex>``.
        """
        return f"{self.settings.job_name_prefix}-{uuid4()}"

    def _render_template(self, substitutions: Dict[str, str]) -> str:
        """Substitute ``{{ KEY }}`` placeholders in the loaded template."""
        rendered = self.job_script_template
        for key, value in substitutions.items():
            rendered = rendered.replace(f"{{{{ {key} }}}}", value)
        return rendered

    def _write_job_script(self, content: str, job_name: str) -> Path:
        """Write rendered ``content`` to a script file and return its path.

        Honors :attr:`ScriptTemplateHPCManagerSettings.job_script_dir`. The
        file is *not* deleted automatically by this method;
        :meth:`create_compute_services` cleans it up after submission unless
        ``keep_job_scripts`` is set.
        """
        directory = self.settings.job_script_dir
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            prefix=f"alchemiscale_{job_name}_",
            dir=str(directory) if directory is not None else None,
            delete=False,
        ) as f:
            f.write(content)
            return Path(f.name)

    @abstractmethod
    def _create_job_script(self) -> Path:
        """Render the template and return a path to the resulting job script."""
