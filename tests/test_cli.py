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
    own = make_job_name("testmgr")
    other = make_job_name("testmgr.bar")
    jobs = [
        {"job_id": "1", "name": own, "state": "RUNNING"},
        {"job_id": "2", "name": other, "state": "RUNNING"},
    ]
    runner = CliRunner()
    args = ["slurm", "show-jobs", "-c", str(slurm_config)]
    with patch.object(SlurmBatchApi, "get_jobs", return_value=jobs):
        mine = runner.invoke(cli, args + ["--mine"])
        everything = runner.invoke(cli, args)
    assert mine.exit_code == 0, mine.output
    assert own in mine.output and other not in mine.output
    assert everything.exit_code == 0, everything.output
    assert own in everything.output and other in everything.output


def test_cleanup_recovers_jobs_before_reporting(slurm_config):
    """The CLI process has no in-memory state, so cleanup first recovers the
    manager's jobs, including recently finished ones."""

    def recover(self, include_finished=False):
        assert include_finished
        self._track("7")
        self._track("8")
        return []

    def clear_failed(self):
        assert self.tracked_jobs == {"7", "8"}
        self._untrack("7")
        return {"7": "TIMEOUT, exit code 0"}

    runner = CliRunner()
    with patch.object(SlurmBatchApi, "recover_jobs", recover), patch.object(
        SlurmBatchApi, "clear_failed_jobs", clear_failed
    ):
        result = runner.invoke(
            cli, ["slurm", "cleanup", "-c", str(slurm_config), "--failed"]
        )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[:2] == [
        "Found 1 failed job(s) of manager 'testmgr':",
        "  7: TIMEOUT, exit code 0",
    ]
    assert "SLURM records are unchanged" in result.output
    assert "it logs each failed job, with its reason" in result.output


def test_cleanup_finds_failed_jobs_through_slurm(slurm_config):
    """End to end through squeue/sacct: cleanup reports this manager's failed
    job with its reason, and not another manager's, without any in-memory
    state."""
    own = [make_job_name("testmgr") for _ in range(3)]
    other = make_job_name("testmgr.bar")
    ts = "2026-11-18T09:30:00"
    run, calls = _fake_slurm(
        squeue_user=f"5|RUNNING|{ts}|{own[0]}\n",
        sacct_user=(
            f"7|FAILED|{ts}|{own[1]}\n"
            f"8|COMPLETED|{ts}|{own[2]}\n"
            f"9|FAILED|{ts}|{other}\n"
        ),
        sacct_tracked=f"7|CANCELLED by 1000|0:15|None|{own[1]}\n",
    )
    runner = CliRunner()
    with patch("subprocess.run", side_effect=run), patch(
        "alchemiscale_hpc.slurm.manager._whoami", return_value="alice"
    ):
        result = runner.invoke(
            cli, ["slurm", "cleanup", "-c", str(slurm_config), "--failed"]
        )
    assert result.exit_code == 0, result.output
    assert "Found 1 failed job(s) of manager 'testmgr':" in result.output
    assert "  7: CANCELLED by 1000, exit code 0, signal 15" in result.output
    # The failed-job query is scoped to this manager's recovered jobs only.
    [scoped] = [c for c in calls if any(a.startswith("--jobs=") for a in c)]
    assert "--jobs=5,7,8" in scoped


def test_cleanup_successful_respects_disabled_setting(slurm_config):
    slurm_config.write_text(
        slurm_config.read_text() + "cleanup_successful_jobs: false\n"
    )
    runner = CliRunner()
    with patch.object(SlurmBatchApi, "recover_jobs", return_value=[]):
        result = runner.invoke(
            cli, ["slurm", "cleanup", "-c", str(slurm_config), "--successful"]
        )
    assert result.exit_code == 0, result.output
    assert "Skipping successful jobs: cleanup_successful_jobs is false" in (
        result.output
    )
    assert "Found" not in result.output


def test_cleanup_reports_successful_jobs(slurm_config):
    def recover(self, include_finished=False):
        self._track("7")
        self._track("8")
        return []

    runner = CliRunner()
    with patch.object(SlurmBatchApi, "recover_jobs", recover), patch.object(
        SlurmBatchApi,
        "_get_completed_jobs",
        return_value=[{"job_id": "8", "state": "COMPLETED", "name": "x"}],
    ):
        result = runner.invoke(
            cli, ["slurm", "cleanup", "-c", str(slurm_config), "--successful"]
        )
    assert result.exit_code == 0, result.output
    assert "Found 1 successful job(s) of manager 'testmgr': 8" in result.output


def test_cleanup_failed_ignores_disabled_setting(slurm_config):
    """``cleanup_failed_jobs`` is ignored (with a warning); the report still
    runs."""
    slurm_config.write_text(
        slurm_config.read_text() + "cleanup_failed_jobs: false\n"
    )
    runner = CliRunner()
    with patch.object(SlurmBatchApi, "recover_jobs", return_value=[]), pytest.warns(
        FutureWarning, match="cleanup_failed_jobs=False is ignored"
    ):
        result = runner.invoke(
            cli, ["slurm", "cleanup", "-c", str(slurm_config), "--failed"]
        )
    assert result.exit_code == 0, result.output
    assert "Found 0 failed job(s) of manager 'testmgr'" in result.output


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
