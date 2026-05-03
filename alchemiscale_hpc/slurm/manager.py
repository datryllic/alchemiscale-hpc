"""SLURM-based compute manager for alchemiscale."""

import subprocess
import tempfile
import re
from pathlib import Path
from uuid import uuid4
from typing import Set, List, Dict

from ..base import HPCManager, HPCBatchApi, JobNotFoundError, JobFailureError
from .settings import SlurmManagerSettings


class SlurmBatchApi(HPCBatchApi):
    """Helper class for interacting with SLURM batch system."""

    def __init__(self, settings: SlurmManagerSettings):
        """Initialize SLURM batch API.

        Parameters
        ----------
        settings : SlurmManagerSettings
            Settings for the SLURM manager.
        """
        super().__init__(settings)
        self.settings: SlurmManagerSettings = settings

    def check_job_health(self):
        """Check if any tracked jobs have failed.

        Raises
        ------
        JobFailureError
            If any tracked job has failed status.
        """
        failed_jobs = self._get_failed_jobs()
        tracked_failed = [job for job in failed_jobs if job["job_id"] in self.tracked_jobs]

        if tracked_failed:
            job_ids = ", ".join([job["job_id"] for job in tracked_failed])
            raise JobFailureError(
                f"Job(s) {job_ids} failed. Check logs and cleanup before restarting the manager."
            )

    def verify_running_jobs(self, server_job_names: Set[str]):
        """Verify that all running jobs are registered with the server.

        Parameters
        ----------
        server_job_names : Set[str]
            Set of job names reported by the alchemiscale server.

        Raises
        ------
        JobNotFoundError
            If a running job is not registered with the server.
        """
        running_jobs = self._get_running_jobs()

        for job in running_jobs:
            job_name = job["name"]
            # Extract base name (before UUID) to match server format
            if job_name.startswith(self.settings.job_name_prefix):
                if job_name not in server_job_names:
                    raise JobNotFoundError(
                        f"Job {job_name} (SLURM ID: {job['job_id']}) not reported by server. "
                        "Possible registration issues."
                    )

    def clear_successful_jobs(self):
        """Track completed jobs for cleanup if configured."""
        if self.settings.cleanup_completed_jobs:
            completed_jobs = self._get_completed_jobs()
            # Remove from tracking set
            for job in completed_jobs:
                self.tracked_jobs.discard(job["job_id"])

    def clear_failed_jobs(self):
        """Remove failed jobs from tracking if configured."""
        if self.settings.cleanup_failed_jobs:
            failed_jobs = self._get_failed_jobs()
            # Remove from tracking set
            for job in failed_jobs:
                self.tracked_jobs.discard(job["job_id"])

    def jobs_pending(self) -> bool:
        """Check if any tracked jobs are pending (not yet running).

        Returns
        -------
        bool
            True if any jobs are in pending state.
        """
        pending_jobs = self._get_pending_jobs()
        return any(job["job_id"] in self.tracked_jobs for job in pending_jobs)

    def get_jobs(self) -> List[Dict[str, str]]:
        """Get all jobs in the queue.

        Returns
        -------
        List[Dict[str, str]]
            List of job dictionaries with keys: job_id, name, state
        """
        cmd = [
            self.settings.query_command,
            "-u", subprocess.check_output(["whoami"]).decode().strip(),
            "-o", "%i,%j,%T",  # job_id, name, state
            "--noheader"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            jobs = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(",")
                    if len(parts) >= 3:
                        jobs.append({
                            "job_id": parts[0],
                            "name": parts[1],
                            "state": parts[2]
                        })
            return jobs
        except subprocess.CalledProcessError:
            # If no jobs, squeue may return non-zero; treat as empty
            return []

    def submit_job(self, script_path: Path) -> str:
        """Submit a job script to SLURM.

        Parameters
        ----------
        script_path : Path
            Path to the job script to submit.

        Returns
        -------
        str
            The SLURM job ID assigned to the submitted job.
        """
        cmd = [self.settings.submit_command, str(script_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse job ID from sbatch output: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            self.tracked_jobs.add(job_id)
            return job_id
        else:
            raise RuntimeError(f"Failed to parse job ID from sbatch output: {result.stdout}")

    def _get_running_jobs(self) -> List[Dict[str, str]]:
        """Get jobs in RUNNING state."""
        all_jobs = self.get_jobs()
        return [job for job in all_jobs if job["state"] == "RUNNING"]

    def _get_pending_jobs(self) -> List[Dict[str, str]]:
        """Get jobs in PENDING state."""
        all_jobs = self.get_jobs()
        return [job for job in all_jobs if job["state"] == "PENDING"]

    def _get_completed_jobs(self) -> List[Dict[str, str]]:
        """Get jobs in COMPLETED state using sacct."""
        cmd = [
            self.settings.accounting_command,
            "-u", subprocess.check_output(["whoami"]).decode().strip(),
            "-s", "COMPLETED",
            "-o", "JobID,JobName,State",
            "--noheader",
            "-X"  # Only show main job, not steps
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            jobs = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        jobs.append({
                            "job_id": parts[0],
                            "name": parts[1],
                            "state": parts[2]
                        })
            return jobs
        except subprocess.CalledProcessError:
            return []

    def _get_failed_jobs(self) -> List[Dict[str, str]]:
        """Get jobs in FAILED, TIMEOUT, CANCELLED, or other error states."""
        cmd = [
            self.settings.accounting_command,
            "-u", subprocess.check_output(["whoami"]).decode().strip(),
            "-s", "FAILED,TIMEOUT,CANCELLED,NODE_FAIL,PREEMPTED",
            "-o", "JobID,JobName,State",
            "--noheader",
            "-X"  # Only show main job, not steps
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            jobs = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        jobs.append({
                            "job_id": parts[0],
                            "name": parts[1],
                            "state": parts[2]
                        })
            return jobs
        except subprocess.CalledProcessError:
            return []


class SlurmManager(HPCManager):
    """Compute manager for SLURM-based HPC systems.

    This manager autoscales compute services by submitting SLURM batch jobs
    based on task availability from the alchemiscale server.
    """

    def __init__(self, settings: SlurmManagerSettings, service_settings_path: Path):
        """Initialize SLURM manager.

        Parameters
        ----------
        settings : SlurmManagerSettings
            Settings for the SLURM manager.
        service_settings_path : Path
            Path to YAML file containing ComputeServiceSettings for services.
        """
        # Initialize parent class (loads settings and template)
        super().__init__(settings=settings, service_settings_path=service_settings_path)

        # Initialize SLURM batch API
        self.batch_api = SlurmBatchApi(self.settings)

    def _create_job_script(self) -> Path:
        """Create a job script from template.

        Returns
        -------
        Path
            Path to the created job script file.
        """
        # Generate unique job name
        job_name = f"{self.settings.job_name_prefix}-{uuid4()}"

        # Prepare template substitutions
        substitutions = {
            "JOB_NAME": job_name,
            "PARTITION": f"#SBATCH --partition={self.settings.partition}" if self.settings.partition else "",
            "ACCOUNT": f"#SBATCH --account={self.settings.account}" if self.settings.account else "",
            "QOS": f"#SBATCH --qos={self.settings.qos}" if self.settings.qos else "",
            "COMPUTE_MANAGER_ID": str(self.compute_manager_id),
        }

        # Fill in template
        job_script_content = self.job_script_template
        for key, value in substitutions.items():
            job_script_content = job_script_content.replace(f"{{{{ {key} }}}}", value)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            prefix=f"alchemiscale_{job_name}_",
            delete=False
        ) as f:
            f.write(job_script_content)
            script_path = Path(f.name)

        return script_path
