"""Base classes and interfaces for HPC batch system managers.

This module provides abstract base classes that define the interface
for HPC-specific compute managers. Each queueing system (SLURM, LSF, PBS, etc.)
should implement these interfaces.
"""

from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Set
from pydantic import Field

from alchemiscale.compute.manager import ComputeManager
from alchemiscale.compute.settings import ComputeManagerSettings


class HPCManagerSettings(ComputeManagerSettings):
    """Base settings class for HPC compute managers.

    Extends ComputeManagerSettings with common HPC-specific configuration.
    """

    job_script_template: Path = Field(
        ...,
        description="Path to batch job script template file."
    )
    job_name_prefix: str = Field(
        "alchemiscale",
        description="Prefix for batch job names. Full name will be <prefix>-<uuid>."
    )
    max_submit_per_cycle: int = Field(
        1,
        description="Maximum number of jobs to submit in a single manager cycle."
    )
    cleanup_completed_jobs: bool = Field(
        True,
        description="If True, track and cleanup completed jobs from accounting system."
    )
    cleanup_failed_jobs: bool = Field(
        True,
        description="If True, track and cleanup failed jobs from accounting system."
    )


class HPCBatchApi:
    """Abstract base class for HPC batch system APIs.

    Each queueing system (SLURM, LSF, PBS) should implement this interface
    to provide a consistent way to interact with the batch system.
    """

    def __init__(self, settings: HPCManagerSettings):
        """Initialize HPC batch API.

        Parameters
        ----------
        settings : HPCManagerSettings
            Settings for the HPC manager.
        """
        self.settings = settings
        self.tracked_jobs: Set[str] = set()

    @abstractmethod
    def check_job_health(self):
        """Check if any tracked jobs have failed.

        Raises
        ------
        JobFailureError
            If any tracked job has failed status.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def clear_successful_jobs(self):
        """Track completed jobs for cleanup if configured."""
        raise NotImplementedError

    @abstractmethod
    def clear_failed_jobs(self):
        """Remove failed jobs from tracking if configured."""
        raise NotImplementedError

    @abstractmethod
    def jobs_pending(self) -> bool:
        """Check if any tracked jobs are pending (not yet running).

        Returns
        -------
        bool
            True if any jobs are in pending state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_jobs(self) -> List[Dict[str, str]]:
        """Get all jobs in the queue.

        Returns
        -------
        List[Dict[str, str]]
            List of job dictionaries. Each dict should contain at minimum:
            - job_id: unique job identifier
            - name: job name
            - state: current job state
        """
        raise NotImplementedError

    @abstractmethod
    def submit_job(self, script_path: Path) -> str:
        """Submit a job script to the batch system.

        Parameters
        ----------
        script_path : Path
            Path to the job script to submit.

        Returns
        -------
        str
            The job ID assigned by the batch system.
        """
        raise NotImplementedError


class HPCManager(ComputeManager):
    """Base class for HPC-specific compute managers.

    This class extends ComputeManager and provides common functionality
    for HPC batch system managers. Subclasses should:
    1. Initialize an HPCBatchApi instance
    2. Implement system-specific job script generation if needed
    """

    batch_api: HPCBatchApi
    job_script_template: str

    def __init__(self, settings: HPCManagerSettings, service_settings_path: Path):
        """Initialize HPC manager.

        Parameters
        ----------
        settings : HPCManagerSettings
            Settings for the HPC manager.
        service_settings_path : Path
            Path to YAML file containing ComputeServiceSettings for services.
        """
        import yaml

        # Load service settings
        with open(service_settings_path, "r") as f:
            service_settings_dict = yaml.safe_load(f)

        # Initialize parent class
        super().__init__(settings=settings, service_settings=service_settings_dict)

        # Load job script template
        with open(self.settings.job_script_template, "r") as f:
            self.job_script_template = f.read()

    def create_compute_services(self, data: dict) -> int:
        """Create compute services by submitting batch jobs.

        This method is called by the base ComputeManager when scaling up is needed.

        Parameters
        ----------
        data : dict
            Data from server instruction containing:
            - compute_service_ids: list of currently active service IDs
            - num_tasks: number of waiting tasks

        Returns
        -------
        int
            Number of new compute services created.
        """
        compute_service_ids = data["compute_service_ids"]
        # Extract job names from service IDs (format: jobname-uuid)
        server_job_names = {csid.split("-")[0] for csid in compute_service_ids}

        self.logger.info("Checking health of batch jobs")
        self.batch_api.check_job_health()

        self.logger.info("Verifying running jobs are registered with server")
        self.batch_api.verify_running_jobs(server_job_names)

        self.logger.info("Cleaning up completed jobs")
        self.batch_api.clear_successful_jobs()

        # Only submit if no jobs are pending
        if not self.batch_api.jobs_pending():
            # Submit up to max_submit_per_cycle jobs
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
            return num_submitted
        else:
            self.logger.info("Skipping job creation, pending jobs exist")
            return 0

    @abstractmethod
    def _create_job_script(self) -> Path:
        """Create a job script from template.

        This method should generate a job script from the template,
        performing any necessary variable substitution.

        Returns
        -------
        Path
            Path to the created job script file.
        """
        raise NotImplementedError


class JobNotFoundError(Exception):
    """Raised when a job is not found in the batch queue."""
    pass


class JobFailureError(Exception):
    """Raised when a batch job has failed."""
    pass
