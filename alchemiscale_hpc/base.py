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
import re
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

import yaml
from pydantic import Field, PrivateAttr, field_validator, model_validator

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


# alchemiscale's ``ComputeIDBase`` name rule (``^[a-zA-Z][a-zA-Z0-9_.:]*$``).
# It applies to the ``ComputeManagerID`` derived from the manager's ``name``,
# and to the ``ComputeServiceID`` each job registers under its job name
# ``{name}.{uuid_hex}``; a name satisfying it yields valid job names too.
_MANAGER_NAME_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_.:]*")


def make_job_name(manager_name: str) -> str:
    """Return a new, unique batch job name for the manager ``manager_name``.

    Job names have the form ``{manager_name}.{uuid_hex}``, with the 32-char
    lowercase hex form of a :func:`uuid.uuid4` (no internal dashes). This
    matches the alchemiscale-k8s convention and keeps the resulting
    ComputeServiceID (``{job_name}-{uuid_hex}``) unambiguously parseable.
    """
    return f"{manager_name}.{uuid4().hex}"


def is_own_job_name(manager_name: str, job_name: str) -> bool:
    """Return True if ``job_name`` was generated for ``manager_name``.

    Matches exactly ``{manager_name}.{32 lowercase hex chars}``, as produced
    by :func:`make_job_name`. A plain prefix check is not enough because
    manager names may contain ``.``: manager ``foo`` must not claim jobs of
    manager ``foo.bar``, which are named ``foo.bar.{uuid_hex}``.
    """
    return (
        re.fullmatch(rf"{re.escape(manager_name)}\.[0-9a-f]{{32}}", job_name)
        is not None
    )


class JobNotFoundError(Exception):
    """Raised when a job is not found in the batch queue."""


class JobFailureError(Exception):
    """Raised when a batch job has failed."""


class HPCManagerSettings(ComputeManagerSettings):
    """Base settings class for HPC compute managers.

    Extends :class:`alchemiscale.compute.settings.ComputeManagerSettings`
    with the minimum HPC-specific configuration that every batch backend
    needs. The per-cycle submission cap (``max_submit_per_cycle``) lives on
    the upstream ``ComputeManagerSettings`` and is inherited from there.

    Batch jobs are named ``{name}.{uuid_hex}``, so the manager's ``name``
    identifies the jobs it submits.
    """

    # Warning emitted because the config set the deprecated
    # ``job_name_prefix`` (to any value, including null), so the manager can
    # repeat it in its own log. See :meth:`_drop_job_name_prefix`.
    _job_name_prefix_warning: Optional[str] = PrivateAttr(None)

    @field_validator("name")
    @classmethod
    def _validate_name_for_hpc(cls, value: str) -> str:
        """Ensure ``name`` can be used to build batch job names.

        The manager names each batch job ``{name}.{uuid4().hex}``, and the
        compute service in that job registers with the job name as its own
        name. Both the ``ComputeManagerID`` derived from ``name`` and the
        ``ComputeServiceID`` derived from the job name must satisfy
        alchemiscale's ID rules, which forbid '-' and require a leading
        letter.
        """
        if not _MANAGER_NAME_PATTERN.fullmatch(value):
            raise ValueError(
                f"name {value!r} is not valid for an HPC compute manager. "
                "It must start with a letter and contain only letters, digits, "
                "underscores ('_'), periods ('.'), and colons (':') (e.g. "
                "'mycluster.gpu'). Batch jobs are named '{name}.{32-character "
                "hex uuid}', and the compute service in each job registers "
                "under its job name; alchemiscale compute IDs forbid '-' and "
                "require a leading letter, so any other name produces compute "
                "services that cannot register."
            )
        return value

    @model_validator(mode="wrap")
    @classmethod
    def _drop_job_name_prefix(cls, data, handler):
        """Backward-compat: ``job_name_prefix`` used to set the job name prefix
        independently of ``name``.

        Job names are now always built from ``name``, so they identify the
        manager that submitted them. alchemiscale rejects a second manager
        registering under a name that is already registered with the same
        server, but managers pointed at different servers can share a name;
        operators should keep names unique per cluster account.

        Existing configs that still set ``job_name_prefix`` continue to load;
        the value is ignored and a :class:`FutureWarning` (shown by default,
        unlike :class:`DeprecationWarning`) names the old and new job names,
        since filters on the old prefix (``squeue``, ``sacct``, ``scancel``)
        stop matching.
        """
        if not isinstance(data, dict) or "job_name_prefix" not in data:
            return handler(data)

        data = dict(data)
        prefix = data.pop("job_name_prefix")
        settings = handler(data)
        message = job_name_prefix_ignored_message(prefix, settings.name)
        settings._job_name_prefix_warning = message
        warnings.warn(message, FutureWarning)
        return settings


def job_name_prefix_ignored_message(prefix, name: str) -> str:
    """Message for a config that still sets the deprecated ``job_name_prefix``."""
    if prefix is None:
        # ``job_name_prefix: null`` never loaded before (the field was a
        # ``str``), so there is no old job name to point to.
        return (
            "job_name_prefix=None is deprecated and ignored; batch jobs are "
            f"named after the manager's name, as '{name}.<uuid_hex>'. Remove "
            "job_name_prefix from the config."
        )
    return (
        f"job_name_prefix={prefix!r} is deprecated and ignored; batch jobs are "
        f"now named after the manager's name, as '{name}.<uuid_hex>' instead "
        f"of '{prefix}.<uuid_hex>'. Update any squeue/sacct/scancel filters "
        "that match the old prefix, and remove job_name_prefix from the config."
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
    * :meth:`recover_jobs` --- rebuild ``tracked_jobs`` from the batch
      system; called once at startup (and by ``cleanup``)

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

    def _track(self, job_id: str, submitted: Optional[datetime] = None) -> None:
        """Record a job and its submission time.

        ``submitted`` defaults to now, as for a job just submitted. Jobs
        recovered from the batch system pass their recorded submission time.
        """
        self.tracked_jobs.add(job_id)
        self._submission_times[job_id] = submitted or datetime.now(tz=timezone.utc)

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

    @abstractmethod
    def recover_jobs(self, include_finished: bool = False) -> List[Dict[str, str]]:
        """Track this manager's jobs found in the batch system.

        ``tracked_jobs`` is held in memory, so it is empty after a restart
        and in each operator CLI invocation. This rebuilds it from the batch
        system: every job of the current user whose name satisfies
        :func:`is_own_job_name` for ``settings.name`` is tracked, with its
        recorded submission time so that
        ``job_registration_grace_period`` still applies. This relies on
        manager names being unique per cluster account.

        Called once by :meth:`HPCManager.start` before the first cycle, and by
        the ``cleanup`` CLI command.

        Parameters
        ----------
        include_finished
            If False, recover only jobs still in the queue (pending, running,
            ...). If True, also recover recently finished (completed or
            failed) jobs; the lookback window is backend-defined.

        Returns
        -------
        list of dict
            The recovered jobs, each with at minimum ``job_id``, ``name``,
            ``state``.
        """

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

        prefix_warning = getattr(settings, "_job_name_prefix_warning", None)
        if prefix_warning is not None:
            self.logger.warning(prefix_warning)

        # Subclasses provide the concrete batch API via _create_batch_api().
        self.batch_api: HPCBatchApi = self._create_batch_api()

    @abstractmethod
    def _create_batch_api(self) -> HPCBatchApi:
        """Construct and return the backend-specific :class:`HPCBatchApi`."""

    def start(self, max_cycles: int | None = None, steal=False):
        """Recover this manager's queued jobs, then run the main loop.

        Without recovery, jobs submitted before a restart would be invisible
        to the cycle: pending jobs would not stop it from submitting more,
        and running jobs would not be checked for registration. Finished
        jobs are not recovered; see :meth:`HPCBatchApi.recover_jobs`.
        """
        recovered = self.batch_api.recover_jobs()
        counts = Counter(job["state"] for job in recovered)
        summary = ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
        self.logger.info(
            f"Recovered {len(recovered)} job(s) from the batch system"
            + (f" ({summary})" if summary else "")
        )
        super().start(max_cycles=max_cycles, steal=steal)


class ScriptTemplateHPCManager(HPCManager):
    """HPC manager that submits jobs by rendering a job-script template.

    Implements the common ``create_compute_services`` workflow shared by the
    SLURM/LSF/PBS backends:

    1. Health-check tracked jobs.
    2. Verify running jobs are registered with the server.
    3. Clear successful jobs (semantics are backend-specific; see
       :meth:`HPCBatchApi.clear_successful_jobs`).
    4. If no jobs are pending, submit up to ``target`` new services. The
       sizing decision (how many to actually submit) is made upstream in
       :meth:`alchemiscale.compute.manager.ComputeManager._compute_jobs_to_create`
       and passed to us as ``target``.

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

    def create_compute_services(self, data: dict, target: int) -> int:
        """Create up to ``target`` compute services by submitting batch jobs.

        Called by the parent :class:`ComputeManager` when scaling up is
        warranted. The cycle has already decided how many services to aim
        for via :meth:`ComputeManager._compute_jobs_to_create`; this method
        just runs the backend-specific health checks and submits scripts.

        If any submitted jobs are still pending in the batch system, this
        method short-circuits to 0 — letting the cluster work through what
        it has before we add more.

        Parameters
        ----------
        data
            Data from the server instruction containing:

            * ``compute_service_ids``: list of currently active service IDs
            * ``num_tasks``: number of waiting tasks
        target
            Upper bound on the number of services to create this cycle.
            Computed by the upstream sizing rule from ``num_tasks``,
            ``max_submit_per_cycle``, remaining capacity, and ``claim_limit``.

        Returns
        -------
        int
            Number of new compute services actually created (``<= target``).
        """
        compute_service_ids = data["compute_service_ids"]

        # Each ComputeServiceID has form "{name}-{uuid_hex}", where `name`
        # cannot itself contain '-'. Strip the trailing hex suffix.
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
        for _ in range(target):
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
        """Return a unique job name of the form ``{name}.{uuid_hex}``.

        See :func:`make_job_name`; :func:`is_own_job_name` recognizes the
        result.
        """
        return make_job_name(self.settings.name)

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
