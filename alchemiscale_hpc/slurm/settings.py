"""Settings for SLURM-based compute management."""

from pydantic import Field

from ..base import ScriptTemplateHPCManagerSettings


class SlurmManagerSettings(ScriptTemplateHPCManagerSettings):
    """Settings for the SLURM compute manager.

    Extends :class:`alchemiscale_hpc.base.ScriptTemplateHPCManagerSettings`
    with SLURM-specific configuration options.
    """

    partition: str | None = Field(
        None,
        description=(
            "SLURM partition to submit jobs to, passed to `sbatch` as "
            "`--partition`. If None, uses the template's directive or the "
            "default partition."
        ),
    )
    account: str | None = Field(
        None,
        description=(
            "SLURM account to charge jobs to, passed to `sbatch` as `--account`. "
            "If None, uses the template's directive or the default account."
        ),
    )
    qos: str | None = Field(
        None,
        description=(
            "Quality of Service for jobs, passed to `sbatch` as `--qos`. If "
            "None, uses the template's directive or the default QOS."
        ),
    )
    submit_command: str = Field(
        "sbatch", description="Command to use for submitting SLURM jobs."
    )
    query_command: str = Field(
        "squeue", description="Command to use for querying SLURM job status."
    )
    accounting_command: str = Field(
        "sacct",
        description="Command to use for querying completed/failed SLURM jobs.",
    )
