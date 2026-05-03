"""Tests for the alchemiscale-hpc CLI."""

from click.testing import CliRunner

from alchemiscale_hpc.cli import cli


def test_top_level_help_lists_all_backends():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "slurm" in result.output
    assert "lsf" in result.output
    assert "pbs" in result.output


def test_slurm_help_lists_all_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["slurm", "--help"])
    assert result.exit_code == 0
    for sub in ("start", "clear-error", "show-jobs", "cleanup"):
        assert sub in result.output


def test_lsf_stub_accepts_standard_options():
    """The LSF stub must accept the same -c/-s options as the SLURM start
    command, so users get a proper "not implemented" message rather than a
    click parse error.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli, ["lsf", "start", "-c", "/tmp/x.yml", "-s", "/tmp/y.yml"]
    )
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_pbs_stub_accepts_cleanup_options():
    """The PBS stub for `cleanup` must tolerate --failed/--completed flags."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["pbs", "cleanup", "-c", "/tmp/x.yml", "--failed", "--completed"]
    )
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_lsf_show_jobs_stub_works():
    runner = CliRunner()
    result = runner.invoke(cli, ["lsf", "show-jobs", "-c", "/tmp/x.yml"])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_unknown_subcommand_fails_cleanly():
    runner = CliRunner()
    result = runner.invoke(cli, ["nonexistent", "start"])
    assert result.exit_code != 0
