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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
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
        description=(
            "Prefix for batch job names. Full name will be "
            "``<prefix>.<uuid_hex>`` (matches the alchemiscale-k8s convention)."
        ),
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
    job_registration_grace_period: int = Field(
        120,
        description=(
            "Seconds to wait after submitting a job before treating it as "
            "'should be registered with the server'. Between SLURM "
            "transitioning a job to RUNNING and the compute service finishing "
            "its registration handshake there is a brief window during which "
            "the job is running but not yet known to the server. Bumping this "
            "value avoids spurious JobNotFoundError on busy clusters; lowering "
            "it makes the manager catch real registration failures faster. "
            "Default 120 seconds."
        ),
    )


class HPCBatchApi(ABC):
    """Abstract base class for HPC batch system APIs.

    Each queueing system (SLURM, LSF, PBS) implements this interface to give
    the manager and CLI a consistent way to interact with the batch system.
    All methods below are abstract; backends must implement every one.

    The methods serve two distinct purposes — both required for a
    production-quality backend, but useful to know which is which when
    implementing or maintaining a new system:

    **Manager-loop primitives** (called every cycle by
    :meth:`ScriptTemplateHPCManager.create_compute_services`; correctness
    of autoscaling depends on these):

    * :meth:`check_job_health` --- raise on tracked-job failure
    * :meth:`verify_running_jobs` --- sanity-check running jobs are
      registered with the alchemiscale server
    * :meth:`clear_successful_jobs` --- per-cycle cleanup of finished jobs
      (backend-defined: in-memory for SLURM-style, real deletion for
      Kubernetes-style)
    * :meth:`jobs_pending` --- gate that lets the cycle skip submitting
      while prior submissions are still queueing
    * :meth:`submit_job` --- the actual submission

    **Operator/diagnostic primitives** (only invoked by the
    ``alchemiscale-hpc <backend> ...`` CLI; the autoscaling loop never
    calls these):

    * :meth:`get_jobs` --- backs ``show-jobs``
    * :meth:`clear_failed_jobs` --- backs ``cleanup --failed``

    The split reflects intent, not optionality. The CLI is part of the
    operator interface for any production deployment, so a backend that
    leaves ``get_jobs`` or ``clear_failed_jobs`` as no-ops is shipping a
    half-finished operator experience.
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
        # Submission timestamps for tracked jobs; populated by ``_track`` and
        # consumed by ``_job_age``. Used to grant newly-submitted jobs a grace
        # period before health checks treat them as "should be registered".
        self._submission_times: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Tracking helpers (concrete; backends should call these rather than
    # mutating ``tracked_jobs`` directly so submission timestamps stay in sync)
    # ------------------------------------------------------------------

    def _track(self, job_id: str) -> None:
        """Record a newly-submitted job and stamp it with the current time."""
        self.tracked_jobs.add(job_id)
        self._submission_times[job_id] = datetime.now(tz=timezone.utc)

    def _untrack(self, job_id: str) -> None:
        """Forget a job (e.g. once it has completed or failed)."""
        self.tracked_jobs.discard(job_id)
        self._submission_times.pop(job_id, None)

    def _job_age(self, job_id: str) -> Optional[timedelta]:
        """Return how long ago ``job_id`` was submitted, or None if unknown."""
        submitted = self._submission_times.get(job_id)
        if submitted is None:
            return None
        return datetime.now(tz=timezone.utc) - submitted

    # ------------------------------------------------------------------
    # Manager-loop primitives
    # ------------------------------------------------------------------
    # Called every cycle by ScriptTemplateHPCManager.create_compute_services.
    # Correctness of autoscaling depends on these.

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
    def jobs_pending(self) -> bool:
        """Return True if any tracked jobs are in a pending (not-yet-running) state."""

    @abstractmethod
    def submit_job(self, script_path: Path) -> str:
        """Submit ``script_path`` to the batch system and return the job ID."""

    # ------------------------------------------------------------------
    # Operator/diagnostic primitives
    # ------------------------------------------------------------------
    # Not called by the manager loop. These back the operator-facing
    # ``alchemiscale-hpc <backend> ...`` CLI commands; backends should
    # still implement them for a complete operator experience.

    @abstractmethod
    def get_jobs(self) -> List[Dict[str, str]]:
        """Get all jobs in the queue. Backs ``alchemiscale-hpc <backend> show-jobs``.

        Returns
        -------
        list of dict
            Each dict contains at minimum ``job_id``, ``name``, ``state``.
        """

    @abstractmethod
    def clear_failed_jobs(self) -> None:
        """Operator-driven cleanup hook for failed jobs. Backs
        ``alchemiscale-hpc <backend> cleanup --failed``.

        Mirrors :meth:`clear_successful_jobs` but for failed jobs. Not called
        on every cycle; only reached after an operator has investigated a
        failure and asked to clear the wreckage.
        """


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
    4. If no jobs are pending, decide how many new services to create as

           min(num_tasks, max_submit_per_cycle, remaining_capacity) // claim_limit

       (with a floor of one if the parent has decided we should scale up at
       all), and submit that many.

    The sizing rule mirrors the alchemiscale-k8s ``K8SManager``: each new
    compute service can claim up to ``claim_limit`` tasks at a time, so to
    cover ``num_tasks`` waiting tasks we need ~``num_tasks / claim_limit``
    services; ``max_submit_per_cycle`` rate-limits how aggressively we ramp
    up; and ``max_compute_services - len(compute_service_ids)`` keeps us
    under the configured capacity ceiling.

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

        The number of jobs submitted is

            ``min(num_tasks, max_submit_per_cycle, remaining_capacity)
            // claim_limit``

        floored at one (the parent only calls us when ``num_tasks > 0`` and
        ``len(compute_service_ids) < max_compute_services``, so there is at
        least one task to serve and one slot to fill). This mirrors the
        sizing logic in ``alchemiscale_k8s.K8SManager.create_compute_services``.

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
        num_tasks = data["num_tasks"]

        # Each ComputeServiceID has form "{name}-{uuid_hex}". Strip only the
        # trailing hex suffix so that service `name` (which may itself contain
        # hyphens, e.g. when ``job_name_prefix`` includes one) is preserved
        # intact.
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

        jobs_to_create = self._compute_jobs_to_create(
            num_tasks=num_tasks,
            num_active_services=len(compute_service_ids),
        )
        if jobs_to_create == 0:
            self.logger.info("No new jobs to create this cycle")
            return 0

        num_submitted = 0
        for _ in range(jobs_to_create):
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

    def _compute_jobs_to_create(self, num_tasks: int, num_active_services: int) -> int:
        """Decide how many new jobs to submit this cycle.

        Mirrors the ``K8SManager`` sizing logic, exposed as a separate method
        so it can be unit-tested without standing up a full manager + batch
        API. See :meth:`create_compute_services` for the full formula.
        """
        remaining_capacity = self.settings.max_compute_services - num_active_services
        # Defensive: if somehow we're already at or over capacity, do nothing.
        if remaining_capacity <= 0 or num_tasks <= 0:
            return 0

        jobs_to_create = min(
            num_tasks,
            self.settings.max_submit_per_cycle,
            remaining_capacity,
        )

        # Each compute service claims up to ``claim_limit`` tasks at a time,
        # so we need fewer services than tasks to cover the queue.
        claim_limit = max(1, self.service_settings.claim_limit)
        jobs_to_create //= claim_limit

        # The parent guarantees we got here because we should scale up. If
        # the divide collapsed to zero (e.g. one task with claim_limit=2),
        # still create one job rather than stalling.
        if jobs_to_create == 0:
            jobs_to_create = 1

        self.logger.info(
            "Sizing: num_tasks=%d, max_submit_per_cycle=%d, "
            "remaining_capacity=%d, claim_limit=%d -> create %d job(s)",
            num_tasks,
            self.settings.max_submit_per_cycle,
            remaining_capacity,
            claim_limit,
            jobs_to_create,
        )
        return jobs_to_create

    def _generate_job_name(self) -> str:
        """Return a unique job name of the form ``{prefix}.{uuid_hex}``.

        Uses the same ``.``-separated format as the alchemiscale-k8s reference
        backend (``f"{name}.{uuid4().hex}"``). The UUID portion is the 32-char
        hex form (no internal dashes), which keeps the resulting
        ComputeServiceID (``{name}-{uuid_hex}``) unambiguously parseable: the
        only ``-`` is the one separating the service name from the trailing
        hex suffix added by ``ComputeServiceID.new_from_name``.
        """
        return f"{self.settings.job_name_prefix}.{uuid4().hex}"

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
