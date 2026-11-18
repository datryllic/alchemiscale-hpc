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
from typing import Dict, List, Optional, Set, Tuple
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
    """Raised when batch jobs keep failing, cycle after cycle."""


class JobSubmissionError(Exception):
    """Raised when batch job submission keeps failing, cycle after cycle."""


class HPCManagerSettings(ComputeManagerSettings):
    """Base settings class for HPC compute managers.

    Extends :class:`alchemiscale.compute.settings.ComputeManagerSettings`
    with the minimum HPC-specific configuration that every batch backend
    needs. The per-cycle submission cap (``max_submit_per_cycle``) lives on
    the upstream ``ComputeManagerSettings`` and is inherited from there.

    Batch jobs are named ``{name}.{uuid_hex}``, so the manager's ``name``
    identifies the jobs it submits.
    """

    job_registration_grace_period: int = Field(
        120,
        description=(
            "Seconds a job may run without being registered with the server. "
            "Between the batch system starting a job and its compute service "
            "finishing its registration handshake, the job is running but not "
            "yet known to the server; the same happens at the end of a job, "
            "between the service deregistering and the job leaving the "
            "running state. Measured from when the job started running, not "
            "from submission. Bumping this value avoids spurious "
            "JobNotFoundError on busy clusters; lowering it makes the manager "
            "catch real registration failures faster. Default 120 seconds."
        ),
    )
    max_consecutive_failed_cycles: int | None = Field(
        5,
        ge=0,
        description=(
            "Failed batch jobs are logged with their reason and untracked "
            "whenever the manager checks its jobs; stop with JobFailureError if "
            "they are found in more than this many checks with no sign of a "
            "healthy job in between. A check that sees a job of this manager "
            "complete, or a compute service of this manager registered with "
            "the server, resets the count; a check with neither failures nor "
            "such a sign leaves it unchanged. Likewise, stop with "
            "JobSubmissionError if every job submission fails in more than "
            "this many cycles that submit while no compute service of this "
            "manager is registered. With a short sleep_interval, raise this "
            "limit so that a brief scheduler outage does not stop the "
            "manager. 0 stops on the first failure; null never stops."
        ),
    )

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
            "Ignored. Failed jobs are always logged once, with their reason, "
            "and dropped from the manager's in-memory tracking, since a failed "
            "job left tracked would be reported again every cycle; batch "
            "system records (e.g. ``sacct``) are kept regardless. Setting it "
            "to false emits a FutureWarning."
        ),
    )

    # Warning emitted because the config set ``cleanup_failed_jobs: false``,
    # so the manager can repeat it in its own log.
    _cleanup_failed_jobs_warning: Optional[str] = PrivateAttr(None)

    @model_validator(mode="after")
    def _warn_cleanup_failed_jobs_ignored(self):
        """Warn when a config sets ``cleanup_failed_jobs: false``, which used
        to keep failed jobs tracked and now has no effect."""
        if "cleanup_failed_jobs" in self.model_fields_set and not (
            self.cleanup_failed_jobs
        ):
            message = (
                "cleanup_failed_jobs=False is ignored; failed jobs are always "
                "logged once, with their reason, and dropped from tracking, "
                "and batch system records are kept regardless. Remove "
                "cleanup_failed_jobs from the config."
            )
            self._cleanup_failed_jobs_warning = message
            warnings.warn(message, FutureWarning)
        return self


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

    * :meth:`clear_failed_jobs` --- untrack failed jobs and report why
      they failed
    * :meth:`find_unregistered_jobs` --- find running jobs that are not
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

    The split reflects intent, not optionality. The CLI is part of the
    operator interface for any production deployment, so a backend that
    leaves ``get_jobs`` as a no-op is shipping a half-finished operator
    experience.
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
        # consumed by ``_job_age``. The registration grace period runs from
        # when a job started; backends fall back to its submission time when
        # the start time is unknown.
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
    def clear_failed_jobs(self) -> Dict[str, str]:
        """Stop tracking failed jobs, returning ``{job_id: failure reason}``.

        Each failed job is returned once: it is untracked, so later calls do
        not report it again. Called every cycle by
        :meth:`HPCManager.handle_finished_jobs`, which logs the reasons and
        decides whether to stop, and once at startup for jobs recovered by
        :meth:`recover_jobs`. Also backs ``cleanup --failed``. For
        Kubernetes-style backends, clearing would also delete the job from
        the batch system.
        """

    @abstractmethod
    def find_unregistered_jobs(
        self, server_job_names: Set[str]
    ) -> List[Dict[str, str]]:
        """Return tracked running jobs that are not registered with the server.

        Jobs that started running less than ``job_registration_grace_period``
        seconds ago are left out, since their compute service may still be
        registering. Whether an unregistered job is an error is decided by
        :meth:`HPCManager.handle_unregistered_jobs`.

        Parameters
        ----------
        server_job_names
            Set of job names reported by the alchemiscale server.

        Returns
        -------
        list of dict
            Each dict contains at minimum ``job_id``, ``name``, ``state``.
        """

    @abstractmethod
    def clear_successful_jobs(self) -> List[str]:
        """Per-cycle cleanup hook for jobs that completed successfully.
        Returns the IDs of the jobs cleared.

        The exact action is backend-defined. For SLURM-style backends this is
        an in-memory operation (drop the IDs from ``tracked_jobs``); for
        Kubernetes-style backends this should actually delete the Job
        objects from the batch system. Either way, this is called
        unconditionally by :meth:`HPCManager.handle_finished_jobs` once per
        cycle, gated only by the ``cleanup_successful_jobs`` setting. A
        cleared job counts as a sign of health there.
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
        the ``cleanup`` CLI command, both with ``include_finished=True``.

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

        for attr in ("_job_name_prefix_warning", "_cleanup_failed_jobs_warning"):
            settings_warning = getattr(settings, attr, None)
            if settings_warning is not None:
                self.logger.warning(settings_warning)

        # Subclasses provide the concrete batch API via _create_batch_api().
        self.batch_api: HPCBatchApi = self._create_batch_api()

        # Checks that found failed jobs since the last sign of health; see
        # handle_finished_jobs.
        self.consecutive_failed_cycles = 0
        # Submitting cycles in which every submission failed, while no
        # service of this manager was registered; see
        # ScriptTemplateHPCManager.create_compute_services.
        self.consecutive_failed_submission_cycles = 0
        # Running jobs found unregistered on the last check, as
        # ``{job_id: (first found, consecutive checks)}``; see
        # handle_unregistered_jobs.
        self._unregistered_jobs: Dict[str, Tuple[datetime, int]] = {}

    @abstractmethod
    def _create_batch_api(self) -> HPCBatchApi:
        """Construct and return the backend-specific :class:`HPCBatchApi`."""

    def start(self, max_cycles: int | None = None, steal=False):
        """Recover this manager's jobs, then run the main loop.

        Without recovery, jobs submitted before a restart would be invisible
        to the cycle: pending jobs would not stop it from submitting more,
        and running jobs would not be checked for registration. If the
        queued jobs cannot be recovered, the manager does not start.

        Recently finished jobs are then recovered separately, see
        :meth:`_report_failures_before_startup`; that is best effort.
        """
        try:
            recovered = self.batch_api.recover_jobs()
        except Exception as e:
            self.logger.error(f"Failed to recover jobs from the batch system: {e}")
            raise
        counts = Counter(job["state"] for job in recovered)
        summary = ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
        self.logger.info(
            f"Recovered {len(recovered)} queued job(s) from the batch system"
            + (f" ({summary})" if summary else "")
        )
        self._report_failures_before_startup()
        super().start(max_cycles=max_cycles, steal=steal)

    def _report_failures_before_startup(self) -> None:
        """Log jobs of this manager that failed recently, e.g. while it was down.

        One WARNING summarizes them, with each job and its reason at INFO.
        They do not count toward ``max_consecutive_failed_cycles``: some were
        already reported before the restart, and a restart should not bring
        the manager closer to stopping. This only informs, so if finished
        jobs cannot be recovered (e.g. accounting is disabled or its
        database is down), a WARNING is logged and startup continues.
        """
        try:
            self.batch_api.recover_jobs(include_finished=True)
        except Exception as e:
            self.logger.warning(
                "Could not recover recently finished jobs from the batch "
                f"system, so failures from before startup are not reported: {e}"
            )
            return
        failed = self.batch_api.clear_failed_jobs()
        if not failed:
            return
        self.logger.warning(
            f"{len(failed)} job(s) of this manager failed recently, before "
            "startup; not counted toward max_consecutive_failed_cycles"
        )
        for job_id, reason in failed.items():
            self.logger.info(f"Batch job {job_id} failed before startup: {reason}")

    def _has_registered_service(self, server_job_names: Set[str]) -> bool:
        """Return True if a compute service of this manager is registered."""
        return any(is_own_job_name(self.settings.name, n) for n in server_job_names)

    def handle_finished_jobs(self, server_job_names: Set[str]) -> Dict[str, str]:
        """Clear successful jobs, and log and untrack failed jobs, raising
        :class:`JobFailureError` if failed jobs are found in more than
        ``max_consecutive_failed_cycles`` checks without a sign of health in
        between. Returns ``{job_id: failure reason}``.

        Each call is one check, so call it once per cycle. A check that sees
        a sign of health, meaning a job of this manager completed or one of
        its compute services is registered with the server, resets the
        count, even if it also found failures. Otherwise, a check with
        failures increments it, and a check without leaves it unchanged.

        So occasional failures in a working fleet (preemption, node failure,
        timeouts) are logged and the manager carries on, while a broken job
        script template or broken settings, whose jobs fail without ever
        registering, stop the manager loudly. Resetting only on a sign of
        health, rather than on any check without failures, matters when jobs
        pend: a check that only sees pending jobs says nothing either way.
        """
        failed = self.batch_api.clear_failed_jobs()
        for job_id, reason in failed.items():
            self.logger.error(f"Batch job {job_id} failed: {reason}")
        completed = self.batch_api.clear_successful_jobs()

        if completed or self._has_registered_service(server_job_names):
            self.consecutive_failed_cycles = 0
        elif failed:
            self.consecutive_failed_cycles += 1
        limit = self.settings.max_consecutive_failed_cycles
        if limit is not None and self.consecutive_failed_cycles > limit:
            raise JobFailureError(
                f"Failed batch jobs found in {self.consecutive_failed_cycles} "
                "cycles with no job completing or registering in between, "
                f"exceeding max_consecutive_failed_cycles={limit}; most recent: "
                f"{', '.join(failed)}. Check the job script template, the job "
                "logs, and the manager and compute service settings."
            )
        return failed

    def handle_unregistered_jobs(self, server_job_names: Set[str]) -> None:
        """Warn about running jobs that are not registered with the server,
        raising :class:`JobNotFoundError` if one stays unregistered.

        A job can be running but unregistered for a while at either end of
        its life: before its compute service registers (left out by
        :meth:`HPCBatchApi.find_unregistered_jobs` during the grace period),
        and after the service deregisters or the server expires a service
        whose heartbeat lapsed, while the job script and epilog finish. So a
        job found unregistered is logged and checked again on the next
        cycle. :class:`JobNotFoundError` is raised once the job has been
        found unregistered on at least two consecutive checks spanning at
        least ``job_registration_grace_period`` seconds. A check that finds
        it registered, or no longer running, forgets it.
        """
        grace = self.settings.job_registration_grace_period
        now = datetime.now(tz=timezone.utc)
        previous, self._unregistered_jobs = self._unregistered_jobs, {}
        for job in self.batch_api.find_unregistered_jobs(server_job_names):
            job_id = job["job_id"]
            first_found, checks = previous.get(job_id, (now, 0))
            checks += 1
            self._unregistered_jobs[job_id] = (first_found, checks)
            elapsed = int((now - first_found).total_seconds())
            if checks > 1 and elapsed >= grace:
                raise JobNotFoundError(
                    f"Job {job['name']} (batch ID: {job_id}) has been running "
                    f"but not registered with the server for {checks} "
                    f"consecutive checks ({elapsed}s). Check that the job "
                    'script template passes `--name "{{ JOB_NAME }}"` to '
                    "`alchemiscale compute synchronous`, and check the job's "
                    "logs. Cancel the job if it is stuck."
                )
            self.logger.warning(
                f"Job {job['name']} (batch ID: {job_id}) is running but not "
                "registered with the server; checking again next cycle"
            )


class ScriptTemplateHPCManager(HPCManager):
    """HPC manager that submits jobs by rendering a job-script template.

    Implements the common ``create_compute_services`` workflow shared by the
    SLURM/LSF/PBS backends:

    1. Clear successful jobs, and log and untrack failed jobs
       (:meth:`HPCManager.handle_finished_jobs`).
    2. Check running jobs are registered with the server
       (:meth:`HPCManager.handle_unregistered_jobs`).
    3. If no jobs are pending, submit up to ``target`` new services. The
       sizing decision (how many to actually submit) is made upstream in
       :meth:`alchemiscale.compute.manager.ComputeManager._compute_jobs_to_create`
       and passed to us as ``target``. If every submission fails in more
       than ``max_consecutive_failed_cycles`` cycles while no service of
       this manager is registered, :class:`JobSubmissionError` is raised.

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

        self.logger.info("Clearing finished batch jobs")
        self.handle_finished_jobs(server_job_names)

        self.logger.info("Verifying running jobs are registered with server")
        self.handle_unregistered_jobs(server_job_names)

        if self.batch_api.jobs_pending():
            self.logger.info("Skipping job creation, pending jobs exist")
            return 0

        num_submitted = 0
        error = None
        for _ in range(target):
            # Only a failure to write the script counts as a failed
            # submission; anything else raised while rendering is a bug.
            try:
                job_script_path = self._create_job_script()
            except OSError as e:
                self.logger.error(f"Failed to write job script: {e}")
                error = e
                break
            try:
                job_id = self.batch_api.submit_job(job_script_path)
                self.logger.info(f"Submitted batch job {job_id}")
                num_submitted += 1
            except Exception as e:
                self.logger.error(f"Failed to submit job: {e}")
                error = e
                break
            finally:
                if not self.settings.keep_job_scripts:
                    try:
                        os.unlink(job_script_path)
                    except OSError:
                        pass
        self._count_failed_submissions(num_submitted, error, server_job_names)
        return num_submitted

    def _count_failed_submissions(
        self,
        num_submitted: int,
        error: Optional[Exception],
        server_job_names: Set[str],
    ) -> None:
        """Count cycles in which every submission failed, raising
        :class:`JobSubmissionError` once there are more than
        ``max_consecutive_failed_cycles``.

        A cycle with at least one successful submission resets the count.
        A cycle that attempts no submission (jobs pending), or whose
        submissions fail while a service of this manager is registered,
        leaves it unchanged: the latter is typically a submit limit
        (e.g. ``MaxSubmitJobsPerUser``) or a brief scheduler outage, not a
        broken configuration.
        """
        if num_submitted:
            self.consecutive_failed_submission_cycles = 0
            return
        if error is None or self._has_registered_service(server_job_names):
            return
        self.consecutive_failed_submission_cycles += 1
        limit = self.settings.max_consecutive_failed_cycles
        if limit is not None and self.consecutive_failed_submission_cycles > limit:
            raise JobSubmissionError(
                "Job submission failed in "
                f"{self.consecutive_failed_submission_cycles} cycles with no "
                "compute service of this manager registered, exceeding "
                f"max_consecutive_failed_cycles={limit}; most recent error: "
                f"{error}. Check the job script template and the manager "
                "settings (e.g. partition, account, QoS, submit command)."
            ) from error

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
