"""SLURM-based compute manager for alchemiscale."""

import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Set

from ..base import (
    HPCBatchApi,
    JobFailureError,
    JobNotFoundError,
    ScriptTemplateHPCManager,
)
from .settings import SlurmManagerSettings

# SLURM job states considered "failed" for health-check purposes.
SLURM_FAILED_STATES = "FAILED,TIMEOUT,CANCELLED,NODE_FAIL,PREEMPTED"

# Field separator for ``squeue``/``sacct`` parsable output. ``%j`` (job name)
# is user-supplied and SLURM does not restrict commas, spaces, or pipes within
# it, so we put the name LAST and split with maxsplit. The chosen separator
# only needs to be absent from job_id (numeric) and state (a fixed set), which
# never contain ``|``.
_FIELD_SEP = "|"
_SQUEUE_FORMAT = f"%i{_FIELD_SEP}%T{_FIELD_SEP}%j"  # job_id, state, name
_SACCT_FORMAT = "JobID,State,JobName"


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

    def check_job_health(self) -> None:
        """Raise :class:`JobFailureError` if any tracked job has failed."""
        # ``_get_failed_jobs`` is already scoped to ``tracked_jobs`` via
        # ``--jobs=``, so every result here is one of ours.
        failed_jobs = self._get_failed_jobs()
        if failed_jobs:
            job_ids = ", ".join(job["job_id"] for job in failed_jobs)
            raise JobFailureError(
                f"Job(s) {job_ids} failed. "
                "Check logs and cleanup before restarting the manager."
            )

    def verify_running_jobs(self, server_job_names: Set[str]) -> None:
        """Verify each running tracked SLURM job is registered with the server.

        Only jobs whose ``job_id`` is in ``self.tracked_jobs`` are validated.
        Other jobs sharing the same ``job_name_prefix`` (e.g. from a second
        manager on the same account or pre-existing user jobs) are ignored.

        Jobs younger than ``settings.job_registration_grace_period`` seconds
        are also skipped to avoid racing the compute service's registration
        handshake.

        Raises
        ------
        JobNotFoundError
            If a tracked, running, past-grace job is not registered.
        """
        grace = self.settings.job_registration_grace_period

        for job in self._get_running_jobs():
            job_id = job["job_id"]
            if job_id not in self.tracked_jobs:
                # Not one of ours — ignore. The previous prefix-only filter
                # would have raised here for any other manager (or pre-existing
                # user job) sharing our prefix.
                continue

            age = self._job_age(job_id)
            if age is not None and age.total_seconds() < grace:
                # Still inside the registration window; the service may not
                # have finished registering yet. Skip this cycle.
                continue

            if job["name"] not in server_job_names:
                raise JobNotFoundError(
                    f"Job {job['name']} (SLURM ID: {job_id}) is running and "
                    f"past its {grace}s registration grace period, but is not "
                    "registered with the server. Check that the SLURM job "
                    'script passes `--name "{{ JOB_NAME }}"` to '
                    "`alchemiscale compute synchronous`."
                )

    def clear_successful_jobs(self) -> None:
        """Drop completed jobs from the in-memory tracking set.

        SLURM ``sacct`` records are immutable history, so there is nothing to
        delete on the cluster side; the manager simply forgets about jobs it
        no longer needs to monitor.
        """
        if not self.settings.cleanup_successful_jobs:
            return
        for job in self._get_completed_jobs():
            self._untrack(job["job_id"])

    def clear_failed_jobs(self) -> None:
        """Drop failed jobs from the in-memory tracking set.

        Same caveat as :meth:`clear_successful_jobs`: this is purely
        in-memory; ``sacct`` records are not modified.
        """
        if not self.settings.cleanup_failed_jobs:
            return
        for job in self._get_failed_jobs():
            self._untrack(job["job_id"])

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
        cmd = [self.settings.submit_command, str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # sbatch output: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Failed to parse job ID from sbatch output: {result.stdout!r}"
            )

        job_id = match.group(1)
        self._track(job_id)
        return job_id

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _squeue_tracked(self, state: str) -> List[Dict[str, str]]:
        """Return tracked jobs in ``state`` via ``squeue --jobs=``.

        Filtering by ``--jobs=<tracked>`` (rather than scanning every job in
        the user's queue) sidesteps the cost of looking at unrelated jobs and
        also means a colocated workload that happens to share our prefix
        cannot interfere.
        """
        if not self.tracked_jobs:
            return []
        cmd = [
            self.settings.query_command,
            "--jobs=" + ",".join(sorted(self.tracked_jobs)),
            "--states=" + state,
            "-o",
            _SQUEUE_FORMAT,
            "--noheader",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            return []
        return self._parse_jobs(result.stdout)

    def _get_running_jobs(self) -> List[Dict[str, str]]:
        return self._squeue_tracked("RUNNING")

    def _get_pending_jobs(self) -> List[Dict[str, str]]:
        return self._squeue_tracked("PENDING")

    def _get_completed_jobs(self) -> List[Dict[str, str]]:
        return self._sacct_query("COMPLETED", self.tracked_jobs)

    def _get_failed_jobs(self) -> List[Dict[str, str]]:
        return self._sacct_query(SLURM_FAILED_STATES, self.tracked_jobs)

    def _sacct_query(
        self, state_filter: str, job_ids: Iterable[str]
    ) -> List[Dict[str, str]]:
        """Run ``sacct`` for ``job_ids`` filtered to ``state_filter``.

        Scoping to specific job IDs via ``--jobs=`` neatly sidesteps the
        ``sacct`` default lookback window (typically "today only"), since
        ``--jobs=`` overrides the time bounds. It also makes the query O(N)
        in tracked jobs rather than O(M) in all the user's history.
        """
        job_ids = list(job_ids)
        if not job_ids:
            # ``sacct`` without ``-j`` would scan all of today's jobs, which is
            # both wasteful and pulls in unrelated work; skip entirely.
            return []

        cmd = [
            self.settings.accounting_command,
            "--jobs=" + ",".join(sorted(job_ids)),
            "-s",
            state_filter,
            "-o",
            _SACCT_FORMAT,
            "--noheader",
            "-X",  # only show main job, not steps
            "-P",  # parsable output: pipe-separated, no padding
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            return []

        return self._parse_jobs(result.stdout)

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


class SlurmManager(ScriptTemplateHPCManager):
    """Compute manager for SLURM-based HPC systems.

    Autoscales compute services by submitting SLURM batch jobs based on
    task availability reported by the alchemiscale server.
    """

    settings: SlurmManagerSettings
    batch_api: SlurmBatchApi

    def _create_batch_api(self) -> SlurmBatchApi:
        return SlurmBatchApi(self.settings)

    def _create_job_script(self) -> Path:
        """Render the SLURM job script template and return its path."""
        job_name = self._generate_job_name()

        substitutions = {
            "JOB_NAME": job_name,
            "PARTITION": (
                f"#SBATCH --partition={self.settings.partition}"
                if self.settings.partition
                else ""
            ),
            "ACCOUNT": (
                f"#SBATCH --account={self.settings.account}"
                if self.settings.account
                else ""
            ),
            "QOS": (f"#SBATCH --qos={self.settings.qos}" if self.settings.qos else ""),
            "COMPUTE_MANAGER_ID": str(self.compute_manager_id),
        }

        rendered = self._render_template(substitutions)
        return self._write_job_script(rendered, job_name)
