"""SLURM-based compute manager for alchemiscale."""

import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, Iterable, List, Optional, Set

from ..base import (
    _TEMPLATE_TOKEN,
    HPCBatchApi,
    ScriptTemplateHPCManager,
    is_own_job_name,
)
from .settings import SlurmManagerSettings

# SLURM job states considered "failed": every terminal state in sacct's JOB
# STATE CODES other than COMPLETED. sacct prints a cancelled job's state as
# ``CANCELLED by <uid>``.
SLURM_FAILED_STATES = (
    "FAILED,TIMEOUT,CANCELLED,NODE_FAIL,PREEMPTED,OUT_OF_MEMORY,BOOT_FAIL,DEADLINE"
)

# Field separator for ``squeue``/``sacct`` parsable output. ``%j`` (job name)
# is user-supplied and SLURM does not restrict commas, spaces, or pipes within
# it, so we put the name LAST and split with maxsplit. The chosen separator
# only needs to be absent from job_id (numeric) and state (a fixed set), which
# never contain ``|``.
_FIELD_SEP = "|"
_SQUEUE_FORMAT = f"%i{_FIELD_SEP}%T{_FIELD_SEP}%j"  # job_id, state, name
_SACCT_FORMAT = "JobID,State,JobName"

# Format for the ``--jobs=``-scoped squeue queries of the manager loop, which
# also need each job's start time (``%S``; actual for running jobs, expected
# for pending ones), since the registration grace period runs from it.
_SQUEUE_TRACKED_FORMAT = f"%i{_FIELD_SEP}%T{_FIELD_SEP}%S{_FIELD_SEP}%j"

# Format for failed jobs: ``ExitCode`` is ``<code>:<signal>``; ``Reason`` is
# the last reason the job was pending (``None`` if there was none). Neither
# contains ``|``, and the name stays last.
_SACCT_FAILED_FORMAT = "JobID,State,ExitCode,Reason,JobName"

# Formats for ``recover_jobs``, which also needs each job's submission time
# (squeue ``%V``, sacct ``Submit``). The name stays last.
_SQUEUE_RECOVER_FORMAT = f"%i{_FIELD_SEP}%T{_FIELD_SEP}%V{_FIELD_SEP}%j"
_SACCT_RECOVER_FORMAT = "JobID,State,Submit,JobName"

# How far back ``recover_jobs(include_finished=True)`` looks for finished jobs
# (sacct ``--starttime``). A week covers a manager that has been down over a
# weekend; older jobs are accounting history rather than anything to act on,
# and a longer window makes sacct scan more of the user's history.
_FINISHED_LOOKBACK = "now-7days"

# Attempts for the user-scoped recovery queries, and the delay before the
# second attempt (doubled after each failure), so a transient slurmctld
# timeout at startup does not stop the manager.
_QUERY_ATTEMPTS = 3
_QUERY_BACKOFF_SECONDS = 2.0


def _query_env() -> Dict[str, str]:
    """Environment for the user-scoped recovery queries.

    squeue and sacct read defaults for their options from ``SQUEUE_*`` and
    ``SACCT_*`` variables (e.g. ``SQUEUE_STATES=all`` would add finished jobs,
    ``SQUEUE_PARTITION`` would hide jobs on other partitions), so these are
    dropped. ``SLURM_TIME_FORMAT=standard`` makes times print as ISO 8601
    (``YYYY-MM-DDTHH:MM:SS``, local time) whatever the user has set.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SQUEUE_", "SACCT_"))
    }
    env["SLURM_TIME_FORMAT"] = "standard"
    return env


def _whoami() -> str:
    """Return the current username, used to scope SLURM queries."""
    return subprocess.check_output(["whoami"], text=True).strip()


class SlurmBatchApi(HPCBatchApi):
    """Helper class for interacting with SLURM batch system."""

    settings: SlurmManagerSettings

    def __init__(self, settings: SlurmManagerSettings):
        super().__init__(settings)

    # ------------------------------------------------------------------
    # HPCBatchApi interface
    # ------------------------------------------------------------------

    def find_unregistered_jobs(
        self, server_job_names: Set[str]
    ) -> List[Dict[str, str]]:
        """Return running tracked SLURM jobs not registered with the server.

        Only jobs whose ``job_id`` is in ``self.tracked_jobs`` are checked.
        Other jobs with similar names (e.g. pre-existing user jobs) are
        ignored.

        Jobs that started running less than
        ``settings.job_registration_grace_period`` seconds ago (squeue
        ``%S``) are left out, to avoid racing the compute service's
        registration handshake. Time spent pending does not count. If the
        start time cannot be parsed, the submission time is used instead.
        """
        grace = self.settings.job_registration_grace_period
        now = datetime.now(tz=timezone.utc)

        unregistered = []
        for job in self._get_running_jobs():
            job_id = job["job_id"]
            if job_id not in self.tracked_jobs:
                # Not one of ours — ignore. The previous prefix-only filter
                # would have raised here for any other manager (or pre-existing
                # user job) sharing our prefix.
                continue

            started = _parse_slurm_time(job["started"])
            age = now - started if started is not None else self._job_age(job_id)
            if age is not None and age.total_seconds() < grace:
                # Still inside the registration window; the service may not
                # have finished registering yet. Skip this cycle.
                continue

            if job["name"] not in server_job_names:
                unregistered.append(
                    {"job_id": job_id, "name": job["name"], "state": job["state"]}
                )
        return unregistered

    def clear_successful_jobs(self) -> List[str]:
        """Drop completed jobs from the in-memory tracking set, returning
        their IDs.

        SLURM ``sacct`` records are immutable history, so there is nothing to
        delete on the cluster side; the manager simply forgets about jobs it
        no longer needs to monitor.
        """
        if not self.settings.cleanup_successful_jobs:
            return []
        cleared = [job["job_id"] for job in self._get_completed_jobs()]
        for job_id in cleared:
            self._untrack(job_id)
        return cleared

    def clear_failed_jobs(self) -> Dict[str, str]:
        """Drop failed jobs from the in-memory tracking set, returning
        ``{job_id: failure reason}``.

        The reason combines sacct's ``State`` (e.g. ``TIMEOUT``,
        ``CANCELLED by 1000``), ``ExitCode``, and ``Reason``. Same caveat as
        :meth:`clear_successful_jobs`: this is purely in-memory; ``sacct``
        records are not modified. Failed jobs are always untracked, since a
        failed job left tracked would be reported again every cycle. sacct's
        ``Reason`` is the last reason the job was pending, so it is labeled
        as such.
        """
        failed = {}
        for job in self._get_failed_jobs():
            failed[job["job_id"]] = job["reason"]
            self._untrack(job["job_id"])
        return failed

    def jobs_pending(self) -> bool:
        """Return True if any tracked jobs are in PENDING state."""
        # ``_get_pending_jobs`` is already scoped to ``tracked_jobs``.
        return bool(self._get_pending_jobs())

    def get_jobs(self) -> List[Dict[str, str]]:
        """Get all jobs in the queue for the current user.

        This returns the unfiltered queue (used by ``alchemiscale-hpc slurm
        show-jobs`` for operator visibility). The internal ``_get_*_jobs``
        helpers used by the manager loop are scoped to ``tracked_jobs`` for
        both correctness and performance.
        """
        cmd = [
            self.settings.query_command,
            "-u",
            _whoami(),
            "-o",
            _SQUEUE_FORMAT,
            "--noheader",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            # If no jobs, squeue may return non-zero; treat as empty.
            return []

        return self._parse_jobs(result.stdout)

    def submit_job(self, script_path: Path) -> str:
        """Submit a job script to SLURM and return the assigned job ID."""
        cmd = [self.settings.submit_command, *self._submit_options(), str(script_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # The exception's own message omits stderr, which says why.
            raise RuntimeError(
                f"{cmd[0]} exited with code {e.returncode}: "
                f"{(e.stderr or '').strip()}"
            ) from e

        # sbatch output: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Failed to parse job ID from sbatch output: {result.stdout!r}"
            )

        job_id = match.group(1)
        self._track(job_id)
        return job_id

    def recover_jobs(self, include_finished: bool = False) -> List[Dict[str, str]]:
        """Track this manager's jobs found in SLURM.

        Queued jobs come from ``squeue -u <user> --all``, in every state
        squeue lists by default (pending, running, suspended, completing,
        ...). ``--all`` includes jobs on hidden partitions, which plain
        ``squeue -u`` omits. With ``include_finished``, jobs that completed or
        failed within ``_FINISHED_LOOKBACK`` also come from ``sacct``. That
        query is not scoped with ``--jobs=``, so it needs an explicit time
        window. ``SQUEUE_*``/``SACCT_*`` environment defaults are ignored (see
        :func:`_query_env`), and each query is retried
        ``_QUERY_ATTEMPTS`` times before failing.

        Only jobs whose name satisfies :func:`is_own_job_name` for
        ``settings.name`` are tracked, so manager ``foo`` does not claim a job
        of manager ``foo.bar``. Submission times come from SLURM; if one
        cannot be parsed, the job is stamped with the current time, which
        grants it the full registration grace period rather than none.

        Raises
        ------
        RuntimeError
            If ``squeue`` or ``sacct`` fails. Proceeding without knowing which
            jobs are ours would let the manager over-submit.
        """
        user = _whoami()
        squeue_cmd = [
            self.settings.query_command,
            "-u",
            user,
            "--all",
            "-o",
            _SQUEUE_RECOVER_FORMAT,
            "--noheader",
        ]
        jobs = self._parse_timed_jobs(self._run_query(squeue_cmd), "submitted")

        if include_finished:
            sacct_cmd = [
                self.settings.accounting_command,
                "-u",
                user,
                "-S",
                _FINISHED_LOOKBACK,
                "-E",
                "now",
                "-s",
                f"COMPLETED,{SLURM_FAILED_STATES}",
                "-o",
                _SACCT_RECOVER_FORMAT,
                "--noheader",
                "-X",  # only show main job, not steps
                "-P",  # parsable output: pipe-separated, no padding
            ]
            jobs += self._parse_timed_jobs(self._run_query(sacct_cmd), "submitted")

        recovered = []
        for job in jobs:
            if not is_own_job_name(self.settings.name, job["name"]):
                continue
            self._track(job["job_id"], _parse_slurm_time(job.pop("submitted")))
            recovered.append(job)
        return recovered

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _submit_options(self) -> List[str]:
        """Return ``sbatch`` options for the partition, account and QoS settings.

        Unset settings are omitted, leaving SLURM's default or any directive
        in the job script. Command-line options take precedence over
        ``#SBATCH`` directives, so a setting overrides the template.
        """
        options = {
            "partition": self.settings.partition,
            "account": self.settings.account,
            "qos": self.settings.qos,
        }
        return [f"--{key}={value}" for key, value in options.items() if value]

    def _squeue_tracked(self, state: str) -> List[Dict[str, str]]:
        """Return tracked jobs in ``state`` via ``squeue --jobs=``.

        Filtering by ``--jobs=<tracked>`` (rather than scanning every job in
        the user's queue) sidesteps the cost of looking at unrelated jobs and
        also means a colocated workload that happens to share our prefix
        cannot interfere. Each job also carries its start time as
        ``started``; see :func:`_query_env` for the time format.
        """
        if not self.tracked_jobs:
            return []
        cmd = [
            self.settings.query_command,
            "--jobs=" + ",".join(sorted(self.tracked_jobs)),
            "--states=" + state,
            "-o",
            _SQUEUE_TRACKED_FORMAT,
            "--noheader",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=_query_env()
            )
        except subprocess.CalledProcessError:
            return []
        return self._parse_timed_jobs(result.stdout, "started")

    def _get_running_jobs(self) -> List[Dict[str, str]]:
        return self._squeue_tracked("RUNNING")

    def _get_pending_jobs(self) -> List[Dict[str, str]]:
        return self._squeue_tracked("PENDING")

    def _get_completed_jobs(self) -> List[Dict[str, str]]:
        return self._parse_jobs(
            self._sacct_query("COMPLETED", self.tracked_jobs, _SACCT_FORMAT)
        )

    def _get_failed_jobs(self) -> List[Dict[str, str]]:
        """Return tracked failed jobs, each with a ``reason``.

        ``sacct -s`` selects jobs that were in a failed state at any time,
        so a job that was preempted and requeued could match while it is
        pending or running again. Only jobs whose current state is a failed
        state are returned.
        """
        failed_states = SLURM_FAILED_STATES.split(",")
        jobs = self._parse_failed_jobs(
            self._sacct_query(
                SLURM_FAILED_STATES, self.tracked_jobs, _SACCT_FAILED_FORMAT
            )
        )
        return [job for job in jobs if job["state"].split(" ", 1)[0] in failed_states]

    def _sacct_query(
        self, state_filter: str, job_ids: Iterable[str], fields: str
    ) -> str:
        """Run ``sacct`` for ``job_ids`` filtered to ``state_filter``, and
        return its parsable output with ``fields`` (``-o``).

        Scoping to specific job IDs via ``--jobs=`` neatly sidesteps the
        ``sacct`` default lookback window (typically "today only"), since
        ``--jobs=`` overrides the time bounds. It also makes the query O(N)
        in tracked jobs rather than O(M) in all the user's history.
        """
        job_ids = list(job_ids)
        if not job_ids:
            # ``sacct`` without ``-j`` would scan all of today's jobs, which is
            # both wasteful and pulls in unrelated work; skip entirely.
            return ""

        cmd = [
            self.settings.accounting_command,
            "--jobs=" + ",".join(sorted(job_ids)),
            "-s",
            state_filter,
            "-o",
            fields,
            "--noheader",
            "-X",  # only show main job, not steps
            "-P",  # parsable output: pipe-separated, no padding
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            return ""

        return result.stdout

    @staticmethod
    def _run_query(cmd: List[str]) -> str:
        """Run a user-scoped SLURM query and return its stdout.

        Unlike the ``--jobs=``-scoped queries, where a non-zero exit can just
        mean the listed jobs have left the queue, a query scoped to the user
        that exits non-zero has failed. It is retried up to
        ``_QUERY_ATTEMPTS`` times, with exponential backoff, before raising.
        """
        delay = _QUERY_BACKOFF_SECONDS
        for attempt in range(1, _QUERY_ATTEMPTS + 1):
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=_query_env()
            )
            if result.returncode == 0:
                return result.stdout
            if attempt < _QUERY_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(
            f"{cmd[0]} exited with code {result.returncode} after "
            f"{_QUERY_ATTEMPTS} attempts: {result.stderr.strip()}"
        )

    @staticmethod
    def _parse_timed_jobs(text: str, time_key: str) -> List[Dict[str, str]]:
        """Parse pipe-separated ``job_id|state|<time>|name`` lines.

        The time is stored under ``time_key`` (e.g. ``submitted``,
        ``started``). As in :meth:`_parse_jobs`, the name is last and may
        contain ``|``.
        """
        jobs: List[Dict[str, str]] = []
        for line in text.strip().splitlines():
            parts = line.split(_FIELD_SEP, 3)
            if len(parts) == 4:
                job_id, state, time_value, name = parts
                jobs.append(
                    {
                        "job_id": job_id,
                        "state": state,
                        time_key: time_value,
                        "name": name,
                    }
                )
        return jobs

    @staticmethod
    def _parse_failed_jobs(text: str) -> List[Dict[str, str]]:
        """Parse pipe-separated ``job_id|state|exit_code|reason|name`` lines.

        Each job gets a ``reason`` summarizing why it failed, e.g.
        ``CANCELLED by 1000, exit code 0, signal 15``. sacct's ``Reason`` is
        the last reason the job was pending, not why it failed. As in
        :meth:`_parse_jobs`, the name is last and may contain ``|``.
        """
        jobs: List[Dict[str, str]] = []
        for line in text.strip().splitlines():
            parts = line.split(_FIELD_SEP, 4)
            if len(parts) != 5:
                continue
            job_id, state, exit_code, reason, name = parts
            details = [state]
            code, _, signal = exit_code.partition(":")
            if code:
                details.append(f"exit code {code}")
            if signal and signal != "0":
                details.append(f"signal {signal}")
            if reason and reason != "None":
                details.append(f"last pending reason {reason}")
            jobs.append(
                {
                    "job_id": job_id,
                    "state": state,
                    "name": name,
                    "reason": ", ".join(details),
                }
            )
        return jobs

    @staticmethod
    def _parse_jobs(text: str) -> List[Dict[str, str]]:
        """Parse pipe-separated ``job_id|state|name`` lines.

        The job name (last column) may itself contain pipes or other
        delimiters since SLURM does not restrict ``%j``; using ``maxsplit=2``
        from the left preserves it intact.
        """
        jobs: List[Dict[str, str]] = []
        for line in text.strip().splitlines():
            if not line:
                continue
            parts = line.split(_FIELD_SEP, 2)
            if len(parts) >= 3:
                jobs.append({"job_id": parts[0], "state": parts[1], "name": parts[2]})
        return jobs


def _parse_slurm_time(value: str) -> Optional[datetime]:
    """Parse a SLURM time (``YYYY-MM-DDTHH:MM:SS``, local) as UTC.

    Returns None for values that are not timestamps (e.g. ``Unknown``).
    """
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    # A naive datetime is taken as local time, which is what SLURM prints.
    return parsed.astimezone(timezone.utc)


class SlurmManager(ScriptTemplateHPCManager):
    """Compute manager for SLURM-based HPC systems.

    Autoscales compute services by submitting SLURM batch jobs based on
    task availability reported by the alchemiscale server.
    """

    settings: SlurmManagerSettings
    batch_api: SlurmBatchApi

    #: Placeholders that earlier templates used for the partition, account
    #: and QoS directives. Those settings are now passed to ``sbatch`` on the
    #: command line, so a template still using them is rejected.
    REMOVED_TEMPLATE_PLACEHOLDERS: ClassVar[FrozenSet[str]] = frozenset(
        {"PARTITION", "ACCOUNT", "QOS"}
    )

    def _create_batch_api(self) -> SlurmBatchApi:
        return SlurmBatchApi(self.settings)

    def _validate_template(self) -> None:
        """Reject removed placeholders, then run the common checks.

        Raises
        ------
        ValueError
            If the template uses any of :attr:`REMOVED_TEMPLATE_PLACEHOLDERS`,
            with any spacing.
        """
        removed = sorted(
            {
                token
                for token in _TEMPLATE_TOKEN.findall(self.job_script_template)
                if token[2:-2].strip() in self.REMOVED_TEMPLATE_PLACEHOLDERS
            }
        )
        if removed:
            raise ValueError(
                f"Job script template {self.settings.job_script_template} uses "
                f"{', '.join(removed)}, which are no longer substituted. The "
                "`partition`, `account` and `qos` settings are now passed to "
                "`sbatch` on the command line; delete these lines from the "
                "template."
            )
        super()._validate_template()

    def _create_job_script(self) -> Path:
        """Render the SLURM job script template and return its path."""
        job_name = self._generate_job_name()

        substitutions = {
            "JOB_NAME": job_name,
            "COMPUTE_MANAGER_ID": str(self.compute_manager_id),
        }

        rendered = self._render_template(substitutions)
        return self._write_job_script(rendered, job_name)
