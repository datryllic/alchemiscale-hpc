"""Tests for the alchemiscale-hpc CLI."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from alchemiscale_hpc.base import make_job_name
from alchemiscale_hpc.cli import _autoload_backends, cli
from alchemiscale_hpc.slurm import SlurmBatchApi

from .test_slurm import _fake_slurm


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
    for sub in ("start", "clear-error", "show-jobs"):
        assert sub in result.output
    assert "cleanup" not in result.output


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


def test_lsf_show_jobs_stub_works():
    runner = CliRunner()
    result = runner.invoke(cli, ["lsf", "show-jobs", "-c", "/tmp/x.yml"])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


@pytest.fixture
def slurm_config(tmp_path):
    config = tmp_path / "manager.yml"
    config.write_text(
        "name: testmgr\n"
        "logfile: null\n"
        "max_compute_services: 1\n"
        f"job_script_template: {tmp_path / 'job-template.sh'}\n"
    )
    return config


def test_show_jobs_mine_shows_only_own_jobs(slurm_config):
    """End to end through squeue/sacct: --mine lists this manager's jobs and
    not another manager's; --finished adds recently finished ones."""
    own = [make_job_name("testmgr") for _ in range(2)]
    other = make_job_name("testmgr.bar")
    ts = "2026-11-18T09:30:00"
    run, calls = _fake_slurm(
        squeue_user=f"5|RUNNING|{ts}|{own[0]}\n6|RUNNING|{ts}|{other}\n",
        sacct_user=f"7|FAILED|{ts}|{own[1]}\n",
    )
    runner = CliRunner()
    args = ["slurm", "show-jobs", "-c", str(slurm_config), "--mine"]
    with patch("subprocess.run", side_effect=run), patch(
        "alchemiscale_hpc.slurm.manager._whoami", return_value="alice"
    ):
        queued = runner.invoke(cli, args)
        finished = runner.invoke(cli, args + ["--finished"])
    assert queued.exit_code == 0, queued.output
    assert own[0] in queued.output
    assert own[1] not in queued.output and other not in queued.output
    assert finished.exit_code == 0, finished.output
    assert own[0] in finished.output and own[1] in finished.output
    assert "FAILED" in finished.output and other not in finished.output


def test_show_jobs_without_mine_shows_all_jobs(slurm_config):
    jobs = [
        {"job_id": "1", "name": make_job_name("testmgr"), "state": "RUNNING"},
        {"job_id": "2", "name": "otherjob", "state": "RUNNING"},
    ]
    runner = CliRunner()
    with patch.object(SlurmBatchApi, "get_jobs", return_value=jobs):
        result = runner.invoke(cli, ["slurm", "show-jobs", "-c", str(slurm_config)])
    assert result.exit_code == 0, result.output
    assert "otherjob" in result.output


def test_show_jobs_finished_requires_mine(slurm_config):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["slurm", "show-jobs", "-c", str(slurm_config), "--finished"]
    )
    assert result.exit_code != 0
    assert "--finished requires --mine" in result.output


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
