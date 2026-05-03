"""Command-line interface for alchemiscale-hpc."""

import click
import yaml
from pathlib import Path


@click.group()
def cli():
    """alchemiscale-hpc: Tools for using alchemiscale with HPC systems.

    Supports multiple queueing systems: SLURM, LSF, PBS, and more.

    Use system-specific subcommands:
        - alchemiscale-hpc slurm ...
        - alchemiscale-hpc lsf ...
        - alchemiscale-hpc pbs ...
    """
    pass


# ============================================================================
# SLURM Commands
# ============================================================================

@cli.group(name="slurm")
def slurm_group():
    """Commands for SLURM queueing system."""
    pass


@slurm_group.command(name="start")
@click.option(
    "-c",
    "--config-file",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML file containing SlurmManagerSettings",
)
@click.option(
    "-s",
    "--service-config-file",
    "service_config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML file containing ComputeServiceSettings",
)
def slurm_start(config_file: Path, service_config_file: Path):
    """Start the SLURM compute manager.

    The manager will continuously monitor task availability and
    autoscale compute services by submitting SLURM batch jobs.

    Example:

        alchemiscale-hpc slurm start -c manager-config.yml -s service-config.yml
    """
    from .slurm import SlurmManager, SlurmManagerSettings

    # Load manager settings
    with open(config_file, "r") as f:
        manager_settings_dict = yaml.safe_load(f)

    manager_settings = SlurmManagerSettings(**manager_settings_dict)

    # Create and start manager
    manager = SlurmManager(
        settings=manager_settings,
        service_settings_path=service_config_file
    )

    # Start the manager (runs until SIGINT or error)
    manager.start()


@slurm_group.command(name="clear-error")
@click.option(
    "-c",
    "--config-file",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML file containing SlurmManagerSettings",
)
@click.option(
    "-s",
    "--service-config-file",
    "service_config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML file containing ComputeServiceSettings",
)
def slurm_clear_error(config_file: Path, service_config_file: Path):
    """Clear error status for a SLURM compute manager.

    Use this if the manager is stuck in ERROR state.

    Example:

        alchemiscale-hpc slurm clear-error -c manager-config.yml -s service-config.yml
    """
    from .slurm import SlurmManager, SlurmManagerSettings

    # Load manager settings
    with open(config_file, "r") as f:
        manager_settings_dict = yaml.safe_load(f)

    manager_settings = SlurmManagerSettings(**manager_settings_dict)

    # Create manager instance
    manager = SlurmManager(
        settings=manager_settings,
        service_settings_path=service_config_file
    )

    # Clear error status
    manager.clear_error()


@slurm_group.command(name="show-jobs")
@click.option(
    "-c",
    "--config-file",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML file containing SlurmManagerSettings",
)
def slurm_show_jobs(config_file: Path):
    """Show all SLURM jobs in the queue.

    Example:

        alchemiscale-hpc slurm show-jobs -c manager-config.yml
    """
    from .slurm import SlurmBatchApi, SlurmManagerSettings

    # Load manager settings
    with open(config_file, "r") as f:
        manager_settings_dict = yaml.safe_load(f)

    manager_settings = SlurmManagerSettings(**manager_settings_dict)

    # Create batch API
    batch_api = SlurmBatchApi(manager_settings)

    # Get and display jobs
    jobs = batch_api.get_jobs()

    if not jobs:
        click.echo("No jobs found in SLURM queue")
        return

    click.echo(f"{'Job ID':<12} {'Name':<30} {'State':<15}")
    click.echo("-" * 57)
    for job in jobs:
        click.echo(f"{job['job_id']:<12} {job['name']:<30} {job['state']:<15}")


@slurm_group.command(name="cleanup")
@click.option(
    "-c",
    "--config-file",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML file containing SlurmManagerSettings",
)
@click.option(
    "--failed",
    is_flag=True,
    help="Clean up failed jobs from tracking",
)
@click.option(
    "--completed",
    is_flag=True,
    help="Clean up completed jobs from tracking",
)
def slurm_cleanup(config_file: Path, failed: bool, completed: bool):
    """Clean up SLURM jobs from tracking.

    Example:

        alchemiscale-hpc slurm cleanup -c manager-config.yml --failed --completed
    """
    from .slurm import SlurmBatchApi, SlurmManagerSettings

    if not failed and not completed:
        click.echo("Please specify at least one of --failed or --completed")
        return

    # Load manager settings
    with open(config_file, "r") as f:
        manager_settings_dict = yaml.safe_load(f)

    manager_settings = SlurmManagerSettings(**manager_settings_dict)

    # Create batch API
    batch_api = SlurmBatchApi(manager_settings)

    if failed:
        click.echo("Cleaning up failed jobs...")
        batch_api.clear_failed_jobs()
        click.echo("Done")

    if completed:
        click.echo("Cleaning up completed jobs...")
        batch_api.clear_successful_jobs()
        click.echo("Done")


# ============================================================================
# LSF Commands (Future)
# ============================================================================

@cli.group(name="lsf")
def lsf_group():
    """Commands for LSF queueing system (coming soon)."""
    pass


@lsf_group.command(name="start")
def lsf_start():
    """Start the LSF compute manager (not yet implemented)."""
    click.echo("LSF support is not yet implemented.")
    click.echo("Contributions welcome! See alchemiscale_hpc/base.py for the interface.")


# ============================================================================
# PBS Commands (Future)
# ============================================================================

@cli.group(name="pbs")
def pbs_group():
    """Commands for PBS/Torque queueing system (coming soon)."""
    pass


@pbs_group.command(name="start")
def pbs_start():
    """Start the PBS compute manager (not yet implemented)."""
    click.echo("PBS support is not yet implemented.")
    click.echo("Contributions welcome! See alchemiscale_hpc/base.py for the interface.")


if __name__ == "__main__":
    cli()
