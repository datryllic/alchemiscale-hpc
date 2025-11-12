"""Command-line interface for alchemiscale-hpc."""

import click
import yaml
from pathlib import Path

from .manager import SlurmManager, SlurmBatchApi
from .settings import SlurmManagerSettings


@click.group()
def cli():
    """alchemiscale-hpc: Tools for using alchemiscale with HPC systems."""
    pass


@cli.group()
def manager():
    """Commands for managing SLURM compute managers."""
    pass


@manager.command(name="start")
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
def manager_start(config_file: Path, service_config_file: Path):
    """Start the SLURM compute manager.

    The manager will continuously monitor task availability and
    autoscale compute services by submitting SLURM batch jobs.

    Example:

        alchemiscale-hpc manager start -c manager_config.yml -s service_config.yml
    """
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


@manager.command(name="clear-error")
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
def manager_clear_error(config_file: Path, service_config_file: Path):
    """Clear error status for a compute manager.

    Use this if the manager is stuck in ERROR state.

    Example:

        alchemiscale-hpc manager clear-error -c manager_config.yml -s service_config.yml
    """
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


@cli.group()
def slurm():
    """Commands for interacting with SLURM directly."""
    pass


@slurm.command(name="show-jobs")
@click.option(
    "-c",
    "--config-file",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML file containing SlurmManagerSettings",
)
def slurm_show_jobs(config_file: Path):
    """Show all tracked SLURM jobs.

    Example:

        alchemiscale-hpc slurm show-jobs -c manager_config.yml
    """
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


@slurm.command(name="cleanup")
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

        alchemiscale-hpc slurm cleanup -c manager_config.yml --failed --completed
    """
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


if __name__ == "__main__":
    cli()
