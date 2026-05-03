"""Command-line interface for alchemiscale-hpc.

The CLI is generated automatically from the backend registry. Each backend
that registers itself via :func:`alchemiscale_hpc.base.register_backend`
gains a ``alchemiscale-hpc <backend> {start,clear-error,show-jobs,cleanup}``
subcommand group, with no further changes required to this module.

Stub backends (currently LSF and PBS) register a placeholder name without a
manager class; their CLI groups exist but every subcommand reports that the
backend is not yet implemented.
"""

from pathlib import Path

import click
import yaml

# Importing the package triggers backend registration.
from . import base as _base

# Backends present as namespace packages but without an implementation. They
# get a CLI group with the standard subcommand signatures so that the docs
# (and any user that copy-pastes a SLURM invocation) work consistently.
_STUB_BACKENDS = {
    "lsf": "LSF",
    "pbs": "PBS/Torque",
}


def _load_settings(config_file: Path, backend: str):
    _, settings_cls, _ = _base.get_backend(backend)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return settings_cls(**config)


def _build_backend_group(backend: str) -> click.Group:
    """Build a click group with the standard subcommands for ``backend``."""

    @click.group(
        name=backend, help=f"Commands for the {backend.upper()} queueing system."
    )
    def group() -> None:
        pass

    config_option = click.option(
        "-c",
        "--config-file",
        "config_file",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help=f"Path to YAML file containing {backend.upper()}ManagerSettings",
    )
    service_option = click.option(
        "-s",
        "--service-config-file",
        "service_config_file",
        type=click.Path(exists=True, path_type=Path),
        required=True,
        help="Path to YAML file containing ComputeServiceSettings",
    )

    @group.command(name="start", help=f"Start the {backend.upper()} compute manager.")
    @config_option
    @service_option
    def start(config_file: Path, service_config_file: Path) -> None:
        manager_cls, _, _ = _base.get_backend(backend)
        settings = _load_settings(config_file, backend)
        manager = manager_cls(
            settings=settings, service_settings_path=service_config_file
        )
        manager.start()

    @group.command(
        name="clear-error",
        help=f"Clear the ERROR status for a {backend.upper()} compute manager.",
    )
    @config_option
    @service_option
    def clear_error(config_file: Path, service_config_file: Path) -> None:
        manager_cls, _, _ = _base.get_backend(backend)
        settings = _load_settings(config_file, backend)
        manager = manager_cls(
            settings=settings, service_settings_path=service_config_file
        )
        manager.clear_error()

    @group.command(
        name="show-jobs",
        help=f"Show all {backend.upper()} jobs in the queue for the current user.",
    )
    @config_option
    def show_jobs(config_file: Path) -> None:
        _, _, batch_api_cls = _base.get_backend(backend)
        settings = _load_settings(config_file, backend)
        batch_api = batch_api_cls(settings)

        jobs = batch_api.get_jobs()
        if not jobs:
            click.echo(f"No jobs found in {backend.upper()} queue")
            return

        click.echo(f"{'Job ID':<12} {'Name':<30} {'State':<15}")
        click.echo("-" * 57)
        for job in jobs:
            click.echo(f"{job['job_id']:<12} {job['name']:<30} {job['state']:<15}")

    @group.command(
        name="cleanup",
        help=f"Clear successful and/or failed {backend.upper()} jobs.",
    )
    @config_option
    @click.option("--failed", is_flag=True, help="Clear failed jobs.")
    @click.option(
        "--successful",
        "--completed",
        "successful",
        is_flag=True,
        help="Clear successful jobs.",
    )
    def cleanup(config_file: Path, failed: bool, successful: bool) -> None:
        if not failed and not successful:
            click.echo("Please specify at least one of --failed or --successful")
            return

        _, _, batch_api_cls = _base.get_backend(backend)
        settings = _load_settings(config_file, backend)
        batch_api = batch_api_cls(settings)

        if failed:
            click.echo("Clearing failed jobs...")
            batch_api.clear_failed_jobs()
            click.echo("Done")

        if successful:
            click.echo("Clearing successful jobs...")
            batch_api.clear_successful_jobs()
            click.echo("Done")

    return group


def _build_stub_group(backend: str, label: str) -> click.Group:
    """Build a click group that mirrors the standard interface but reports
    that ``backend`` is not yet implemented.

    The stub subcommands accept and ignore any options/arguments, so
    invocations like ``alchemiscale-hpc lsf start -c x.yml -s y.yml`` work
    consistently with the SLURM equivalents.
    """

    @click.group(
        name=backend,
        help=f"Commands for the {label} queueing system (coming soon).",
    )
    def group() -> None:
        pass

    def _make_stub(name: str) -> click.Command:
        @click.command(
            name=name,
            help=f"{name.replace('-', ' ').capitalize()} ({label}, not yet implemented).",
            context_settings={
                "ignore_unknown_options": True,
                "allow_extra_args": True,
            },
        )
        @click.pass_context
        def stub(ctx: click.Context) -> None:
            click.echo(f"{label} support is not yet implemented.")
            click.echo(
                "Contributions welcome! See alchemiscale_hpc/base.py for the interface."
            )
            ctx.exit(1)

        return stub

    for subcommand in ("start", "clear-error", "show-jobs", "cleanup"):
        group.add_command(_make_stub(subcommand))

    return group


@click.group()
def cli() -> None:
    """alchemiscale-hpc: Tools for using alchemiscale with HPC systems.

    Each supported queueing system is exposed as a subcommand group, e.g.

    \b
        alchemiscale-hpc slurm start -c manager.yml -s service.yml
        alchemiscale-hpc slurm show-jobs -c manager.yml
    """


def _register_groups() -> None:
    """Attach a click group for each registered backend, and stubs for known
    placeholder backends.
    """
    registered = set(_base.list_backends())
    for backend in registered:
        cli.add_command(_build_backend_group(backend))
    for backend, label in _STUB_BACKENDS.items():
        if backend in registered:
            continue
        cli.add_command(_build_stub_group(backend, label))


# Lazily import known backend packages so they can self-register.
def _autoload_backends() -> None:
    import importlib

    for backend in ("slurm", "lsf", "pbs"):
        try:
            importlib.import_module(f"alchemiscale_hpc.{backend}")
        except ImportError:
            # A backend might be intentionally absent; skip silently.
            pass


_autoload_backends()
_register_groups()


if __name__ == "__main__":
    cli()
