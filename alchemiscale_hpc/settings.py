"""Settings for SLURM-based compute management."""

from pathlib import Path
from pydantic import Field
from alchemiscale.compute.settings import ComputeManagerSettings


class SlurmManagerSettings(ComputeManagerSettings):
    """Settings for the SLURM compute manager.

    This extends the base ComputeManagerSettings with SLURM-specific
    configuration options.
    """

    job_script_template: Path = Field(
        ...,
        description="Path to SLURM job script template file."
    )
    partition: str | None = Field(
        None,
        description="SLURM partition to submit jobs to. If None, uses default partition."
    )
    account: str | None = Field(
        None,
        description="SLURM account to charge jobs to. If None, uses default account."
    )
    qos: str | None = Field(
        None,
        description="Quality of Service specification for jobs. If None, uses default QOS."
    )
    job_name_prefix: str = Field(
        "alchemiscale",
        description="Prefix for SLURM job names. Full name will be <prefix>-<uuid>."
    )
    submit_command: str = Field(
        "sbatch",
        description="Command to use for submitting SLURM jobs."
    )
    cancel_command: str = Field(
        "scancel",
        description="Command to use for canceling SLURM jobs."
    )
    query_command: str = Field(
        "squeue",
        description="Command to use for querying SLURM job status."
    )
    accounting_command: str = Field(
        "sacct",
        description="Command to use for querying completed/failed SLURM jobs."
    )
    max_submit_per_cycle: int = Field(
        1,
        description="Maximum number of jobs to submit in a single cycle."
    )
    cleanup_completed_jobs: bool = Field(
        True,
        description="If True, track and cleanup completed jobs from accounting system."
    )
    cleanup_failed_jobs: bool = Field(
        True,
        description="If True, track and cleanup failed jobs from accounting system."
    )
