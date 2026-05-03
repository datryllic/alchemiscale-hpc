"""SLURM-based compute manager for alchemiscale."""

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set

from ..base import (
    HPCBatchApi,
    JobFailureError,
    JobNotFoundError,
    ScriptTemplateHPCManager,
)
from .settings import SlurmManagerSettings

# SLURM job states considered "failed" for health-check purposes.
SLURM_FAILED_STATES = "FAILED,TIMEOUT,CANCELLED,NODE_FAIL,PREEMPTED"


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
        failed_jobs = self._get_failed_jobs()
        tracked_failed = [
            job for job in failed_jobs if job["job_id"] in self.tracked_jobs
        ]

        if tracked_failed:
            job_ids = ", ".join(job["job_id"] for job in tracked_failed)
            raise JobFailureError(
                f"Job(s) {job_ids} failed. "
                "Check logs and cleanup before restarting the manager."
            )

    def verify_running_jobs(self, server_job_names: Set[str]) -> None:
        """Verify each running SLURM job has a corresponding registered service.

        ``server_job_names`` is the set of compute service ``name`` values
        reported by the alchemiscale server (extracted from
        ``ComputeServiceID`` strings of the form ``"{name}-{uuid_hex}"``).
        SLURM jobs submitted by this manager use the same value as both their
        SLURM job name and the compute service ``--name``, so each running
        SLURM job's name should appear in ``server_job_names``.

        Raises
        ------
        JobNotFoundError
            If a running job submitted by this manager is not registered with
            the server.
        """
        running_jobs = self._get_running_jobs()

        for job in running_jobs:
            job_name = job["name"]
            if not job_name.startswith(self.settings.job_name_prefix):
                # Not one of ours; skip.
                continue
            if job_name not in server_job_names:
                raise JobNotFoundError(
                    f"Job {job_name} (SLURM ID: {job['job_id']}) is running "
                    "but not registered with the server. "
                    "Check that the SLURM job script passes "
                    '`--name "{{ JOB_NAME }}"` to '
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
            self.tracked_jobs.discard(job["job_id"])

    def clear_failed_jobs(self) -> None:
        """Drop failed jobs from the in-memory tracking set.

        Same caveat as :meth:`clear_successful_jobs`: this is purely
        in-memory; ``sacct`` records are not modified.
        """
        if not self.settings.cleanup_failed_jobs:
            return
        for job in self._get_failed_jobs():
            self.tracked_jobs.discard(job["job_id"])

    def jobs_pending(self) -> bool:
        """Return True if any tracked jobs are in PENDING state."""
        pending = self._get_pending_jobs()
        return any(job["job_id"] in self.tracked_jobs for job in pending)

    def get_jobs(self) -> List[Dict[str, str]]:
        """Get all jobs in the queue for the current user."""
        cmd = [
            self.settings.query_command,
            "-u",
            _whoami(),
            "-o",
            "%i,%j,%T",  # job_id, name, state
            "--noheader",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            # If no jobs, squeue may return non-zero; treat as empty.
            return []

        return self._parse_csv_jobs(result.stdout)

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
        self.tracked_jobs.add(job_id)
        return job_id

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _get_running_jobs(self) -> List[Dict[str, str]]:
        return [job for job in self.get_jobs() if job["state"] == "RUNNING"]

    def _get_pending_jobs(self) -> List[Dict[str, str]]:
        return [job for job in self.get_jobs() if job["state"] == "PENDING"]

    def _get_completed_jobs(self) -> List[Dict[str, str]]:
        return self._sacct_query("COMPLETED")

    def _get_failed_jobs(self) -> List[Dict[str, str]]:
        return self._sacct_query(SLURM_FAILED_STATES)

    def _sacct_query(self, state_filter: str) -> List[Dict[str, str]]:
        """Run ``sacct`` filtered by ``state_filter`` and parse the output."""
        cmd = [
            self.settings.accounting_command,
            "-u",
            _whoami(),
            "-s",
            state_filter,
            "-o",
            "JobID,JobName,State",
            "--noheader",
            "-X",  # only show main job, not steps
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            return []

        return self._parse_whitespace_jobs(result.stdout)

    @staticmethod
    def _parse_csv_jobs(text: str) -> List[Dict[str, str]]:
        """Parse ``squeue --noheader -o '%i,%j,%T'`` output."""
        jobs: List[Dict[str, str]] = []
        for line in text.strip().splitlines():
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                jobs.append({"job_id": parts[0], "name": parts[1], "state": parts[2]})
        return jobs

    @staticmethod
    def _parse_whitespace_jobs(text: str) -> List[Dict[str, str]]:
        """Parse ``sacct`` whitespace-delimited output."""
        jobs: List[Dict[str, str]] = []
        for line in text.strip().splitlines():
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                jobs.append({"job_id": parts[0], "name": parts[1], "state": parts[2]})
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
