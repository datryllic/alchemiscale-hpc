"""Tests for the alchemiscale-hpc CLI."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from alchemiscale_hpc.cli import _autoload_backends, cli


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


# ----------------------------------------------------------------------
# _autoload_backends error handling (review issue #3)
# ----------------------------------------------------------------------


def test_autoload_swallows_genuinely_missing_subpackage():
    """If a backend subpackage doesn't exist at all, _autoload_backends
    should silently skip it — that's the legitimate "backend not installed"
    case the swallow is for.
    """
    import importlib

    def missing_subpackage(name):
        # Simulate every backend being absent at the top level.
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    with patch.object(importlib, "import_module", side_effect=missing_subpackage):
        # Should NOT raise: every loop iteration raises with name == the
        # subpackage being imported.
        _autoload_backends()


def test_autoload_propagates_nested_import_errors():
    """If a backend subpackage IS present but raises a *different* missing
    module (e.g. a typo or missing dependency in its own code), the error
    must propagate. Otherwise a real bug looks like "backend silently absent".
    """
    import importlib

    def boom(name):
        # Top-level subpackage import succeeds in finding the package, then
        # raises a ModuleNotFoundError for a nested name.
        raise ModuleNotFoundError(
            "No module named 'some_missing_dep'", name="some_missing_dep"
        )

    with patch.object(importlib, "import_module", side_effect=boom):
        with pytest.raises(ModuleNotFoundError, match="some_missing_dep"):
            _autoload_backends()


def test_autoload_propagates_arbitrary_exceptions():
    """Non-ImportError exceptions (SyntaxError, RuntimeError, etc.) from a
    backend's import must also propagate, not be silently swallowed."""
    import importlib

    def boom(name):
        raise RuntimeError("backend init blew up")

    with patch.object(importlib, "import_module", side_effect=boom):
        with pytest.raises(RuntimeError, match="blew up"):
            _autoload_backends()
