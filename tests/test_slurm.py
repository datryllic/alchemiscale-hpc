"""Tests for the SLURM batch API and manager."""

import re
import subprocess
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from alchemiscale.storage.models import ComputeManagerInstruction

from alchemiscale_hpc import (
    HPCBatchApi,
    JobFailureError,
    JobNotFoundError,
    JobSubmissionError,
)
from alchemiscale_hpc.base import is_own_job_name, make_job_name
from alchemiscale_hpc.slurm import (
    SlurmBatchApi,
    SlurmManager,
    SlurmManagerSettings,
)
from alchemiscale_hpc.slurm.manager import _parse_slurm_time


@pytest.fixture
def slurm_settings(tmp_path: Path) -> SlurmManagerSettings:
    template = tmp_path / "job-template.sh"
    template.write_text(
        "#!/bin/bash\n#SBATCH --job-name={{ JOB_NAME }}\n"
        "echo manager={{ COMPUTE_MANAGER_ID }}\n"
    )
    return SlurmManagerSettings(
        name="testmgr",
        logfile=None,
        max_compute_services=2,
        job_script_template=template,
        partition="testpart",
        account="testacct",
        # Tests assume jobs are immediately past grace; override the default.
        job_registration_grace_period=0,
    )


def _track(api: SlurmBatchApi, job_id: str, age_seconds: float = 9999) -> None:
    """Track ``job_id`` and back-date its submission so it's past grace."""
    api._track(job_id)
    api._submission_times[job_id] = datetime.now(tz=timezone.utc) - timedelta(
        seconds=age_seconds
    )


def test_slurm_batch_api_is_concrete(slurm_settings: SlurmManagerSettings):
    """SlurmBatchApi implements every abstract method on HPCBatchApi."""
    api = SlurmBatchApi(slurm_settings)
    assert isinstance(api, HPCBatchApi)
    # tracked_jobs initialized empty
    assert api.tracked_jobs == set()
    assert api._submission_times == {}


def test_slurm_settings_extends_script_template_base():
    """SlurmManagerSettings exposes the inherited fields and SLURM-specific ones."""
    fields = set(SlurmManagerSettings.model_fields)
    # Inherited from upstream ComputeManagerSettings:
    assert "max_submit_per_cycle" in fields
    # Removed in favor of ``name`` (see HPCManagerSettings):
    assert "job_name_prefix" not in fields
    # Inherited from ScriptTemplateHPCManagerSettings:
    assert {
        "job_script_template",
        "cleanup_successful_jobs",
        "cleanup_failed_jobs",
        "keep_job_scripts",
        "job_script_dir",
        "job_registration_grace_period",
    } <= fields
    # SLURM-specific:
    assert {
        "partition",
        "account",
        "qos",
        "submit_command",
        "query_command",
        "accounting_command",
    } <= fields
    # cancel_command was removed when cancel_job left the interface.
    assert "cancel_command" not in fields


def _settings_kwargs(tmp_path: Path, **overrides) -> dict:
    template = tmp_path / "job-template.sh"
    template.write_text(
        "#!/bin/bash\n#SBATCH --job-name={{ JOB_NAME }}\n"
        "echo manager={{ COMPUTE_MANAGER_ID }}\n"
    )
    kwargs = dict(
        name="testmgr",
        logfile=None,
        max_compute_services=1,
        job_script_template=template,
    )
    kwargs.update(overrides)
    return kwargs


# The naming rule itself is alchemiscale's; these only check that it is
# applied when settings load, with the HPC explanation.
@pytest.mark.parametrize("name", ["mycluster.gpu", "Site:queue.v2"])
def test_settings_accepts_valid_name(tmp_path: Path, name: str):
    settings = SlurmManagerSettings(**_settings_kwargs(tmp_path, name=name))
    assert settings.name == name


@pytest.mark.parametrize("name", ["my-slurm-manager", "1cluster"])
def test_settings_rejects_invalid_name(tmp_path: Path, name: str):
    """Names that cannot form a valid ComputeServiceID are rejected up front."""
    with pytest.raises(ValueError, match="alchemiscale compute IDs forbid '-'"):
        SlurmManagerSettings(**_settings_kwargs(tmp_path, name=name))


def test_settings_job_name_prefix_is_deprecated_and_ignored(tmp_path: Path):
    """Configs that still set ``job_name_prefix`` load, with a visible warning.

    The prefix is ignored, even an invalid one, since job names are built
    from ``name``. The warning is a FutureWarning (shown by default) naming
    the ignored prefix and the new job names, since operators' filters on
    the old prefix stop matching.
    """
    with pytest.warns(FutureWarning) as record:
        settings = SlurmManagerSettings(
            **_settings_kwargs(tmp_path, job_name_prefix="my-experiment")
        )
    assert not hasattr(settings, "job_name_prefix")
    [warning] = record
    message = str(warning.message)
    assert "job_name_prefix='my-experiment' is deprecated and ignored" in message
    assert "'testmgr.<uuid_hex>'" in message


def test_settings_without_job_name_prefix_does_not_warn(tmp_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        settings = SlurmManagerSettings(**_settings_kwargs(tmp_path))
    assert settings._job_name_prefix_warning is None


def test_settings_null_job_name_prefix_still_warns(tmp_path: Path, caplog):
    """An explicit ``job_name_prefix: null`` is still a deprecated setting."""
    with pytest.warns(FutureWarning) as record:
        settings = SlurmManagerSettings(
            **_settings_kwargs(tmp_path, job_name_prefix=None)
        )
    [warning] = record
    message = str(warning.message)
    assert "job_name_prefix=None is deprecated and ignored" in message
    assert "'testmgr.<uuid_hex>'" in message
    assert "None.<uuid_hex>" not in message

    with caplog.at_level("WARNING", logger="AlchemiscaleComputeManager"):
        _build_manager(settings, tmp_path)
    assert "job_name_prefix=None is deprecated and ignored" in caplog.text


def test_manager_logs_ignored_job_name_prefix(tmp_path: Path, caplog):
    """The manager repeats the job_name_prefix warning in its own log."""
    with pytest.warns(FutureWarning):
        settings = SlurmManagerSettings(
            **_settings_kwargs(tmp_path, job_name_prefix="alchemiscale")
        )
    with caplog.at_level("WARNING", logger="AlchemiscaleComputeManager"):
        _build_manager(settings, tmp_path)
    assert "job_name_prefix='alchemiscale' is deprecated and ignored" in caplog.text


def test_is_own_job_name_matches_generated_names():
    assert is_own_job_name("foo", make_job_name("foo"))
    assert is_own_job_name("foo.bar", make_job_name("foo.bar"))


@pytest.mark.parametrize(
    "job_name",
    [
        # Another manager whose name extends ours with a ".".
        f"foo.bar.{uuid4().hex}",
        # Another manager whose name extends ours without a ".".
        f"foobar.{uuid4().hex}",
        # Not a uuid4().hex suffix: uppercase, too short, too long.
        f"foo.{uuid4().hex.upper()}",
        "foo.deadbeef",
        f"foo.{uuid4().hex}0",
        # Regex metacharacters in the name are matched literally.
        f"fooX{uuid4().hex}",
        "foo",
    ],
)
def test_is_own_job_name_rejects_other_names(job_name: str):
    assert not is_own_job_name("foo", job_name)


def test_parse_jobs_handles_empty():
    assert SlurmBatchApi._parse_jobs("") == []
    assert SlurmBatchApi._parse_jobs("\n\n") == []


def test_parse_jobs_parses_pipe_output():
    out = "12345|RUNNING|alchemiscale-abcd\n67890|PENDING|otherjob\n"
    parsed = SlurmBatchApi._parse_jobs(out)
    assert parsed == [
        {"job_id": "12345", "state": "RUNNING", "name": "alchemiscale-abcd"},
        {"job_id": "67890", "state": "PENDING", "name": "otherjob"},
    ]


def test_parse_jobs_preserves_separators_in_job_name():
    """Job name is the last column; embedded ``|`` (or commas, spaces) survive.

    This guards against the previous CSV implementation, where a comma in a
    user-supplied job name would silently corrupt parsing.
    """
    out = "12345|RUNNING|weird,name|with|pipes and spaces\n"
    parsed = SlurmBatchApi._parse_jobs(out)
    assert parsed == [
        {
            "job_id": "12345",
            "state": "RUNNING",
            "name": "weird,name|with|pipes and spaces",
        }
    ]


# ----------------------------------------------------------------------
# clear_failed_jobs: failure reasons from sacct
# ----------------------------------------------------------------------


def test_parse_failed_jobs_builds_reasons():
    """State, ExitCode (code:signal) and Reason are combined; the job name
    stays last and may contain the separator."""
    out = (
        "1|FAILED|1:0|None|a.b\n"
        "2|CANCELLED by 1000|0:15|None|c\n"
        "3|TIMEOUT|0:0|None|d\n"
        "4|NODE_FAIL|0:0|ReqNodeNotAvail|weird|name\n"
        "5|OUT_OF_MEMORY|0:125|None|e\n"
        "garbage\n"
    )
    assert [
        (job["job_id"], job["state"], job["name"], job["reason"])
        for job in SlurmBatchApi._parse_failed_jobs(out)
    ] == [
        ("1", "FAILED", "a.b", "FAILED, exit code 1"),
        ("2", "CANCELLED by 1000", "c", "CANCELLED by 1000, exit code 0, signal 15"),
        ("3", "TIMEOUT", "d", "TIMEOUT, exit code 0"),
        (
            "4",
            "NODE_FAIL",
            "weird|name",
            "NODE_FAIL, exit code 0, last pending reason ReqNodeNotAvail",
        ),
        ("5", "OUT_OF_MEMORY", "e", "OUT_OF_MEMORY, exit code 0, signal 125"),
    ]


def test_clear_failed_jobs_returns_reasons_and_untracks(
    slurm_settings: SlurmManagerSettings,
):
    """Failed jobs come from a ``--jobs=``-scoped sacct query including the
    exit code and reason; each is returned once, then untracked."""
    api = SlurmBatchApi(slurm_settings)
    for jid in ("1", "2", "3"):
        _track(api, jid)
    run, calls = _fake_slurm(
        sacct_tracked=(
            "1|CANCELLED by 1000|0:15|None|x\n"
            "2|PREEMPTED|0:0|None|y\n"
        )
    )
    with patch("subprocess.run", side_effect=run):
        assert api.clear_failed_jobs() == {
            "1": "CANCELLED by 1000, exit code 0, signal 15",
            "2": "PREEMPTED, exit code 0",
        }
    assert api.tracked_jobs == {"3"}
    [cmd] = calls
    assert "--jobs=1,2,3" in cmd
    assert "JobID,State,ExitCode,Reason,JobName" in cmd
    states = set(cmd[cmd.index("-s") + 1].split(","))
    # Every terminal state other than COMPLETED, per sacct's JOB STATE CODES.
    assert states == {
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "NODE_FAIL",
        "PREEMPTED",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "DEADLINE",
    }


def test_clear_failed_jobs_ignores_requeued_jobs(
    slurm_settings: SlurmManagerSettings,
):
    """``sacct -s PREEMPTED`` can match a job that was requeued and is
    pending again; it is still ours to track, not a failure."""
    api = SlurmBatchApi(slurm_settings)
    _track(api, "1")
    run, _ = _fake_slurm(sacct_tracked="1|PENDING|0:0|None|x\n")
    with patch("subprocess.run", side_effect=run):
        assert api.clear_failed_jobs() == {}
    assert api.tracked_jobs == {"1"}


def test_clear_failed_jobs_ignores_cleanup_setting(
    slurm_settings: SlurmManagerSettings,
):
    """A failed job left tracked would be reported again every cycle, so
    ``cleanup_failed_jobs: false`` does not keep it tracked."""
    api = SlurmBatchApi(slurm_settings.model_copy(update={"cleanup_failed_jobs": False}))
    _track(api, "1")
    run, _ = _fake_slurm(sacct_tracked="1|FAILED|1:0|None|x\n")
    with patch("subprocess.run", side_effect=run):
        assert api.clear_failed_jobs() == {"1": "FAILED, exit code 1"}
    assert api.tracked_jobs == set()


# ----------------------------------------------------------------------
# find_unregistered_jobs
# ----------------------------------------------------------------------


def _running(job_id: str, name: str, started: str = "Unknown") -> dict:
    """A running job as returned by ``_get_running_jobs``."""
    return {"job_id": job_id, "name": name, "state": "RUNNING", "started": started}


def test_find_unregistered_jobs_recognizes_registered(
    slurm_settings: SlurmManagerSettings,
):
    """A tracked SLURM job whose name matches a server-known service passes."""
    api = SlurmBatchApi(slurm_settings)
    _track(api, "1")
    job_name = f"{slurm_settings.name}.deadbeef"
    with patch.object(
        api, "_get_running_jobs", return_value=[_running("1", job_name)]
    ):
        assert api.find_unregistered_jobs({job_name}) == []


def test_find_unregistered_jobs_returns_unregistered(
    slurm_settings: SlurmManagerSettings,
):
    """A tracked, past-grace, running job not in server_job_names is returned."""
    api = SlurmBatchApi(slurm_settings)
    _track(api, "1")
    job_name = f"{slurm_settings.name}.deadbeef"
    with patch.object(
        api, "_get_running_jobs", return_value=[_running("1", job_name)]
    ):
        assert api.find_unregistered_jobs(set()) == [
            {"job_id": "1", "name": job_name, "state": "RUNNING"}
        ]


def test_find_unregistered_jobs_ignores_untracked_jobs(
    slurm_settings: SlurmManagerSettings,
):
    """Jobs we did not submit are ignored, even if they share our name.

    Regression: the previous implementation filtered by prefix only, which
    meant a second manager (or a stray user job) with the same prefix would
    trip the check.
    """
    api = SlurmBatchApi(slurm_settings)
    # tracked_jobs intentionally empty
    with patch.object(
        api,
        "_get_running_jobs",
        return_value=[_running("1", f"{slurm_settings.name}.fromothermanager")],
    ):
        assert api.find_unregistered_jobs(set()) == []


def test_find_unregistered_jobs_grace_runs_from_start_time(
    slurm_settings: SlurmManagerSettings, denver_tz
):
    """The grace period runs from when the job started running (squeue
    ``%S``), not from submission: a job that pended for an hour and has just
    started gets the full grace period."""
    settings = slurm_settings.model_copy(update={"job_registration_grace_period": 600})
    api = SlurmBatchApi(settings)
    _track(api, "1", age_seconds=3600)
    job_name = f"{settings.name}.old"
    now = datetime.now(tz=timezone.utc)
    for started_ago, unregistered in ((10, False), (900, True)):
        started = _local(now - timedelta(seconds=started_ago))
        with patch.object(
            api, "_get_running_jobs", return_value=[_running("1", job_name, started)]
        ):
            assert bool(api.find_unregistered_jobs(set())) is unregistered


def test_find_unregistered_jobs_unknown_start_falls_back_to_submit_time(
    slurm_settings: SlurmManagerSettings,
):
    """If squeue's start time does not parse, the submission time is used."""
    settings = slurm_settings.model_copy(update={"job_registration_grace_period": 60})
    api = SlurmBatchApi(settings)
    job_name = f"{settings.name}.x"
    for submitted_ago, unregistered in ((10, False), (120, True)):
        _track(api, "1", age_seconds=submitted_ago)
        with patch.object(
            api, "_get_running_jobs", return_value=[_running("1", job_name, "N/A")]
        ):
            assert bool(api.find_unregistered_jobs(set())) is unregistered


# ----------------------------------------------------------------------
# jobs_pending / clear_*_jobs
# ----------------------------------------------------------------------


def test_jobs_pending_reflects_get_pending_jobs(
    slurm_settings: SlurmManagerSettings,
):
    """``_get_pending_jobs`` is already tracked-scoped, so any result counts."""
    api = SlurmBatchApi(slurm_settings)
    _track(api, "12345")
    with patch.object(api, "_get_pending_jobs", return_value=[]):
        assert api.jobs_pending() is False
    with patch.object(
        api,
        "_get_pending_jobs",
        return_value=[{"job_id": "12345", "name": "x", "state": "PENDING"}],
    ):
        assert api.jobs_pending() is True


def test_clear_successful_jobs_removes_from_set_and_times(
    slurm_settings: SlurmManagerSettings,
):
    api = SlurmBatchApi(slurm_settings)
    for jid in ("1", "2", "3"):
        _track(api, jid)
    with patch.object(
        api,
        "_get_completed_jobs",
        return_value=[
            {"job_id": "1", "name": "x", "state": "COMPLETED"},
            {"job_id": "2", "name": "y", "state": "COMPLETED"},
        ],
    ):
        api.clear_successful_jobs()
    assert api.tracked_jobs == {"3"}
    # _untrack should also clear the submission timestamps.
    assert "1" not in api._submission_times
    assert "2" not in api._submission_times
    assert "3" in api._submission_times


def test_clear_failed_jobs_removes_from_set_and_times(
    slurm_settings: SlurmManagerSettings,
):
    api = SlurmBatchApi(slurm_settings)
    for jid in ("1", "2", "3"):
        _track(api, jid)
    with patch.object(
        api,
        "_get_failed_jobs",
        return_value=[
            {"job_id": "1", "name": "x", "state": "FAILED", "reason": "FAILED"}
        ],
    ):
        assert api.clear_failed_jobs() == {"1": "FAILED"}
    assert api.tracked_jobs == {"2", "3"}
    assert "1" not in api._submission_times


def test_clear_successful_jobs_respects_setting(tmp_path: Path):
    template = tmp_path / "job-template.sh"
    template.write_text("#!/bin/bash\n#SBATCH --job-name={{ JOB_NAME }}\n")
    settings = SlurmManagerSettings(
        name="testmgr",
        logfile=None,
        max_compute_services=2,
        job_script_template=template,
        cleanup_successful_jobs=False,
    )
    api = SlurmBatchApi(settings)
    _track(api, "1")
    with patch.object(
        api,
        "_get_completed_jobs",
        return_value=[{"job_id": "1", "name": "x", "state": "COMPLETED"}],
    ):
        api.clear_successful_jobs()
    # Setting disabled -> nothing removed
    assert api.tracked_jobs == {"1"}


# ----------------------------------------------------------------------
# Tracked-scoped query helpers
# ----------------------------------------------------------------------


def test_squeue_tracked_skips_when_no_tracked_jobs(
    slurm_settings: SlurmManagerSettings,
):
    """No tracked jobs -> no subprocess call at all (perf + correctness)."""
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run") as mock_run:
        assert api._squeue_tracked("RUNNING") == []
    mock_run.assert_not_called()


def test_squeue_tracked_passes_jobs_filter(slurm_settings: SlurmManagerSettings):
    """``--jobs=`` is built from ``tracked_jobs`` and ``--states=`` from arg."""
    api = SlurmBatchApi(slurm_settings)
    _track(api, "100")
    _track(api, "200")
    fake = type(
        "R",
        (),
        {
            "stdout": "100|RUNNING|2026-11-18T09:30:00|x\n",
            "stderr": "",
            "returncode": 0,
        },
    )
    with patch("subprocess.run", return_value=fake) as mock_run:
        jobs = api._squeue_tracked("RUNNING")
    call_args = mock_run.call_args[0][0]
    assert "--jobs=100,200" in call_args
    assert "--states=RUNNING" in call_args
    # Start time (``%S``) before the name, as ISO 8601 whatever the user set.
    assert "%i|%T|%S|%j" in call_args
    assert mock_run.call_args.kwargs["env"]["SLURM_TIME_FORMAT"] == "standard"
    assert jobs == [
        {
            "job_id": "100",
            "state": "RUNNING",
            "started": "2026-11-18T09:30:00",
            "name": "x",
        }
    ]


def test_sacct_query_skips_when_no_tracked_jobs(
    slurm_settings: SlurmManagerSettings,
):
    """No tracked jobs -> no sacct call. Avoids scanning the user's history."""
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run") as mock_run:
        assert api._sacct_query("COMPLETED", set(), "JobID") == ""
    mock_run.assert_not_called()


def test_sacct_query_uses_jobs_filter_and_parsable_output(
    slurm_settings: SlurmManagerSettings,
):
    """sacct is called with ``--jobs=`` (sidesteps lookback) and ``-P``
    (pipe-separated parsable output)."""
    api = SlurmBatchApi(slurm_settings)
    fake = type("R", (), {"stdout": "", "stderr": "", "returncode": 0})
    with patch("subprocess.run", return_value=fake) as mock_run:
        api._sacct_query("COMPLETED", {"100", "200"}, "JobID")
    call_args = mock_run.call_args[0][0]
    assert "--jobs=100,200" in call_args
    assert "-P" in call_args  # parsable / pipe-separated
    assert "-X" in call_args  # main jobs only, no steps


# ----------------------------------------------------------------------
# submit_job
# ----------------------------------------------------------------------


def test_submit_job_parses_sbatch_output_and_tracks(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    api = SlurmBatchApi(slurm_settings)
    fake = type(
        "R",
        (),
        {"stdout": "Submitted batch job 99999\n", "stderr": "", "returncode": 0},
    )
    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    with patch("subprocess.run", return_value=fake):
        job_id = api.submit_job(script)
    assert job_id == "99999"
    assert "99999" in api.tracked_jobs
    # submit_job should also stamp the submission time via _track.
    assert "99999" in api._submission_times


def test_submit_job_passes_partition_account_qos(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Set options are passed to sbatch before the script; unset ones are not."""
    api = SlurmBatchApi(slurm_settings)
    fake = type(
        "R",
        (),
        {"stdout": "Submitted batch job 1\n", "stderr": "", "returncode": 0},
    )
    script = tmp_path / "script.sh"
    with patch("subprocess.run", return_value=fake) as run:
        api.submit_job(script)
    assert run.call_args.args[0] == [
        "sbatch",
        "--partition=testpart",
        "--account=testacct",
        str(script),
    ]


def test_submit_job_raises_on_unparseable_output(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    api = SlurmBatchApi(slurm_settings)
    fake = type("R", (), {"stdout": "what is this", "stderr": "", "returncode": 0})
    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    with patch("subprocess.run", return_value=fake):
        with pytest.raises(RuntimeError, match="Failed to parse job ID"):
            api.submit_job(script)


def test_no_cancel_job_method(slurm_settings: SlurmManagerSettings):
    """Regression: cancel_job was dropped because no caller (framework or
    CLI) ever uses it. The k8s reference implementation likewise has no
    per-job cancel — only bulk clear_failed_jobs / clear_successful_jobs.
    """
    api = SlurmBatchApi(slurm_settings)
    assert not hasattr(api, "cancel_job")


# ----------------------------------------------------------------------
# Manager-level integration tests
# ----------------------------------------------------------------------


def _build_manager(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
) -> SlurmManager:
    """Helper: build a SlurmManager with the parent-class init bypassed."""
    service_config = tmp_path / "service.yml"
    service_config.write_text(
        "api_url: http://example.com\n"
        "identifier: id\n"
        "key: k\n"
        "name: svc\n"
        "shared_basedir: /tmp\n"
        "scratch_basedir: /tmp\n"
    )
    # Bypass network/registration in ComputeManager.__init__ by patching the
    # client constructor it triggers.
    # ``autospec`` so upstream signature drift in e.g. ``get_instruction`` or
    # ``update_status`` fails these tests.
    with patch(
        "alchemiscale.compute.manager.AlchemiscaleComputeManagerClient", autospec=True
    ):
        mgr = SlurmManager(
            settings=slurm_settings, service_settings_path=service_config
        )
    # Instance attribute set in the real client's __init__; autospec can't see it.
    mgr.client.api_url = "http://example.com"
    return mgr


def test_slurm_manager_create_batch_api_factory(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """SlurmManager wires up its own SlurmBatchApi automatically."""
    mgr = _build_manager(slurm_settings, tmp_path)
    assert isinstance(mgr.batch_api, SlurmBatchApi)


def test_slurm_manager_render_template(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Template substitution renders all expected placeholders."""
    mgr = _build_manager(slurm_settings, tmp_path)
    script_path = mgr._create_job_script()
    try:
        content = script_path.read_text()
        assert slurm_settings.name in content
        # partition/account/qos are passed to sbatch, not rendered
        assert "--partition" not in content
        # COMPUTE_MANAGER_ID substituted
        assert "{{ COMPUTE_MANAGER_ID }}" not in content
    finally:
        script_path.unlink(missing_ok=True)


def test_slurm_manager_create_compute_services_cleans_up_scripts(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """After submission the rendered script is removed unless keep_job_scripts."""
    mgr = _build_manager(slurm_settings, tmp_path)

    submitted_paths = []

    def _fake_submit(path: Path) -> str:
        submitted_paths.append(path)
        return "12345"

    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={}
    ), patch.object(
        mgr.batch_api, "find_unregistered_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "clear_successful_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(mgr.batch_api, "submit_job", side_effect=_fake_submit):
        n = mgr.create_compute_services(
            {"compute_service_ids": [], "num_tasks": 1}, target=1
        )
    assert n == 1
    assert submitted_paths
    # The script should have been removed after submission.
    assert not submitted_paths[0].exists()


def test_slurm_manager_keep_job_scripts(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    slurm_settings = slurm_settings.model_copy(
        update={"keep_job_scripts": True, "job_script_dir": tmp_path / "scripts"}
    )
    mgr = _build_manager(slurm_settings, tmp_path)

    submitted_paths = []

    def _fake_submit(path: Path) -> str:
        submitted_paths.append(path)
        return "12345"

    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={}
    ), patch.object(
        mgr.batch_api, "find_unregistered_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "clear_successful_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(mgr.batch_api, "submit_job", side_effect=_fake_submit):
        mgr.create_compute_services(
            {"compute_service_ids": [], "num_tasks": 1}, target=1
        )
    # The script should still exist.
    assert submitted_paths[0].exists()
    submitted_paths[0].unlink()


# ----------------------------------------------------------------------
# create_compute_services target handling
# ----------------------------------------------------------------------
#
# The sizing math itself (min(num_tasks, max_submit_per_cycle, capacity)
# // claim_limit, with floor-to-1) lives upstream in
# ``alchemiscale.compute.manager.ComputeManager._compute_jobs_to_create``
# and is unit-tested there. The tests below verify only that we honor
# the ``target`` we receive and short-circuit for backend-specific reasons.


def test_create_compute_services_honors_target(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Given target=N, submit exactly N scripts."""
    mgr = _build_manager(slurm_settings, tmp_path)

    submitted = []

    def _fake_submit(path: Path) -> str:
        submitted.append(path)
        return f"id-{len(submitted)}"

    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={}
    ), patch.object(
        mgr.batch_api, "find_unregistered_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "clear_successful_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(mgr.batch_api, "submit_job", side_effect=_fake_submit):
        n = mgr.create_compute_services(
            {"compute_service_ids": [], "num_tasks": 100}, target=3
        )
    assert n == 3
    assert len(submitted) == 3


def test_create_compute_services_skips_when_pending(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Backend gate: if any submitted jobs are still pending, do nothing
    even when ``target`` says to scale up."""
    mgr = _build_manager(slurm_settings, tmp_path)
    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={}
    ), patch.object(
        mgr.batch_api, "find_unregistered_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "clear_successful_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "jobs_pending", return_value=True
    ), patch.object(mgr.batch_api, "submit_job") as submit:
        n = mgr.create_compute_services(
            {"compute_service_ids": [], "num_tasks": 5}, target=5
        )
    assert n == 0
    submit.assert_not_called()


def test_maintain_compute_services_strips_uuid_suffix_correctly(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """server_job_names should strip the trailing ComputeServiceID hex suffix.

    With the standard ``{name}.{uuid_hex}`` job name format there is exactly
    one ``-`` in the resulting ``ComputeServiceID`` (the one separating the
    name from the random hex suffix added by
    ``ComputeServiceID.new_from_name``), so ``rsplit("-", 1)[0]`` recovers
    the original name cleanly.
    """
    mgr = _build_manager(slurm_settings, tmp_path)

    service_name = f"{slurm_settings.name}.deadbeef0123456789abcdef01234567"
    service_id = f"{service_name}-cafebabe0123456789abcdef01234567"

    captured = {}

    def _capture(server_job_names):
        captured["names"] = server_job_names
        return []

    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={}
    ), patch.object(
        mgr.batch_api, "find_unregistered_jobs", side_effect=_capture
    ), patch.object(
        mgr.batch_api, "clear_successful_jobs", return_value=[]
    ):
        mgr.maintain_compute_services(
            {"compute_service_ids": [service_id], "num_tasks": 0}
        )

    assert captured["names"] == {service_name}


def test_create_compute_services_does_not_run_maintenance(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """``create_compute_services`` only submits; housekeeping lives in
    ``maintain_compute_services``."""
    mgr = _build_manager(slurm_settings, tmp_path)
    with patch.object(mgr, "maintain_compute_services") as maintain, patch.object(
        mgr, "handle_finished_jobs"
    ) as finished, patch.object(
        mgr, "handle_unregistered_jobs"
    ) as unregistered, patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", return_value="1"
    ):
        mgr.create_compute_services(
            {"compute_service_ids": [], "num_tasks": 1}, target=1
        )
    maintain.assert_not_called()
    finished.assert_not_called()
    unregistered.assert_not_called()


# ----------------------------------------------------------------------
# cycle(): maintenance runs every OK/SKIP cycle, independent of scale-up
# ----------------------------------------------------------------------


def test_cycle_runs_maintenance_with_no_tasks(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """An OK instruction with ``num_tasks=0`` never reaches
    ``create_compute_services`` upstream, but maintenance still runs."""
    mgr = _build_manager(slurm_settings, tmp_path)
    service_name = f"{slurm_settings.name}.deadbeef0123456789abcdef01234567"
    service_id = f"{service_name}-cafebabe0123456789abcdef01234567"
    mgr.client.get_instruction.return_value = (
        ComputeManagerInstruction.OK,
        {"compute_service_ids": [service_id], "num_tasks": 0},
    )

    with patch.object(mgr, "handle_finished_jobs") as finished, patch.object(
        mgr, "handle_unregistered_jobs"
    ) as unregistered, patch.object(
        mgr, "create_compute_services"
    ) as create:
        assert mgr.cycle() is True

    finished.assert_called_once_with({service_name})
    unregistered.assert_called_once_with({service_name})
    create.assert_not_called()
    mgr.client.update_status.assert_called_once()


def test_cycle_runs_maintenance_at_capacity(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """At ``max_compute_services`` upstream skips scale-up; maintenance runs."""
    mgr = _build_manager(slurm_settings, tmp_path)
    ids = [f"svc.{i:032x}-{i:032x}" for i in range(slurm_settings.max_compute_services)]
    mgr.client.get_instruction.return_value = (
        ComputeManagerInstruction.OK,
        {"compute_service_ids": ids, "num_tasks": 10},
    )

    with patch.object(mgr, "maintain_compute_services") as maintain, patch.object(
        mgr, "create_compute_services"
    ) as create:
        assert mgr.cycle() is True

    maintain.assert_called_once()
    create.assert_not_called()


def test_cycle_runs_maintenance_before_scale_up(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """On a scale-up cycle, maintenance runs once, before submission."""
    mgr = _build_manager(slurm_settings, tmp_path)
    mgr.client.get_instruction.return_value = (
        ComputeManagerInstruction.OK,
        {"compute_service_ids": [], "num_tasks": 1},
    )

    calls = []
    with patch.object(
        mgr,
        "maintain_compute_services",
        side_effect=lambda data: calls.append("maintain"),
    ), patch.object(
        mgr,
        "create_compute_services",
        side_effect=lambda data, target: calls.append("create") or 0,
    ):
        assert mgr.cycle() is True

    assert calls == ["maintain", "create"]


def test_cycle_scale_up_runs_each_primitive_once(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """End-to-end scale-up cycle with the real ``create_compute_services``:
    each maintenance primitive runs exactly once, then the job is submitted."""
    mgr = _build_manager(slurm_settings, tmp_path)
    mgr.client.get_instruction.return_value = (
        ComputeManagerInstruction.OK,
        {"compute_service_ids": [], "num_tasks": 1},
    )

    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={}
    ) as failed, patch.object(
        mgr.batch_api, "find_unregistered_jobs", return_value=[]
    ) as unregistered, patch.object(
        mgr.batch_api, "clear_successful_jobs", return_value=[]
    ) as completed, patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", return_value="1"
    ) as submit:
        assert mgr.cycle() is True

    failed.assert_called_once_with()
    unregistered.assert_called_once_with(set())
    completed.assert_called_once_with()
    submit.assert_called_once()


class _FakeClient:
    """Plain (non-mock) client, to observe how ``cycle`` patches it."""

    api_url = "http://example.com"

    def get_instruction(self, scopes, protocols, compute_manager_id):
        return ComputeManagerInstruction.OK, {
            "compute_service_ids": [],
            "num_tasks": 0,
        }

    def update_status(self, compute_manager_id, status, detail=None, saturation=None):
        pass


def test_cycle_restores_client_get_instruction(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """``cycle`` leaves no ``get_instruction`` instance attribute behind, so
    later cycles resolve the method afresh from the client class."""
    mgr = _build_manager(slurm_settings, tmp_path)
    client = _FakeClient()
    mgr.client = client

    with patch.object(mgr, "maintain_compute_services") as maintain:
        assert mgr.cycle() is True
        assert "get_instruction" not in vars(client)
        assert mgr.cycle() is True
        assert "get_instruction" not in vars(client)

    assert maintain.call_count == 2


def test_cycle_runs_maintenance_on_skip(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """SKIP still carries ``compute_service_ids``; housekeeping still runs."""
    mgr = _build_manager(slurm_settings, tmp_path)
    data = {"compute_service_ids": []}
    mgr.client.get_instruction.return_value = (ComputeManagerInstruction.SKIP, data)

    with patch.object(mgr, "maintain_compute_services") as maintain, patch.object(
        mgr, "create_compute_services"
    ) as create:
        assert mgr.cycle() is True

    maintain.assert_called_once_with(data)
    create.assert_not_called()


def test_cycle_skips_maintenance_on_shutdown(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    mgr = _build_manager(slurm_settings, tmp_path)
    mgr.client.get_instruction.return_value = (
        ComputeManagerInstruction.SHUTDOWN,
        {"message": "bye"},
    )

    with patch.object(mgr, "maintain_compute_services") as maintain:
        assert mgr.cycle() is False

    maintain.assert_not_called()
    mgr.client.update_status.assert_not_called()


def test_cycle_propagates_maintenance_errors(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """A failing finished-job check aborts the cycle before sizing or
    submission, and the client's ``get_instruction`` is restored afterwards."""
    mgr = _build_manager(slurm_settings, tmp_path)
    get_instruction = mgr.client.get_instruction
    get_instruction.return_value = (
        ComputeManagerInstruction.OK,
        {"compute_service_ids": [], "num_tasks": 1},
    )

    with patch.object(
        mgr, "handle_finished_jobs", side_effect=JobFailureError("boom")
    ), patch.object(mgr, "create_compute_services") as create:
        with pytest.raises(JobFailureError, match="boom"):
            mgr.cycle()

    create.assert_not_called()
    mgr.client.update_status.assert_not_called()
    assert mgr.client.get_instruction is get_instruction


def test_generate_job_name_format(slurm_settings: SlurmManagerSettings, tmp_path: Path):
    """Generated job names are ``{name}.{32-char hex}`` (k8s-style)."""
    mgr = _build_manager(slurm_settings, tmp_path)

    name = mgr._generate_job_name()

    prefix = slurm_settings.name
    # Exactly one separator "." between the manager name and the uuid hex.
    assert name.startswith(f"{prefix}.")
    suffix = name[len(prefix) + 1 :]
    # 32-char lowercase hex (uuid4().hex).
    assert re.fullmatch(r"[0-9a-f]{32}", suffix), suffix
    # Each call yields a fresh uuid.
    assert name != mgr._generate_job_name()


# ----------------------------------------------------------------------
# recover_jobs: rebuilding tracked_jobs after a restart
# ----------------------------------------------------------------------


class _Result:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _fake_slurm(
    squeue_user: str = "",
    sacct_user: str = "",
    tracked: dict = None,
    sacct_tracked: str = "",
    sacct_completed: str = "",
):
    """Fake ``subprocess.run`` for squeue/sacct.

    ``squeue_user``/``sacct_user`` answer user-scoped queries (recovery);
    ``tracked`` maps a ``--states=`` value to the answer for ``--jobs=``-scoped
    squeue queries, and ``sacct_tracked`` answers ``--jobs=``-scoped sacct
    queries for failed jobs (the manager loop), ``sacct_completed`` those
    for completed jobs. Every call is recorded in ``calls``.
    """
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "squeue" and "-u" in cmd:
            return _Result(squeue_user)
        if cmd[0] == "squeue":
            state = next(a for a in cmd if a.startswith("--states="))
            return _Result((tracked or {}).get(state.split("=", 1)[1], ""))
        if cmd[0] == "sacct" and "-u" in cmd:
            return _Result(sacct_user)
        if cmd[0] == "sacct" and cmd[cmd.index("-s") + 1] == "COMPLETED":
            return _Result(sacct_completed)
        if cmd[0] == "sacct":
            return _Result(sacct_tracked)
        return _Result()

    return run, calls


@pytest.fixture
def whoami():
    with patch("alchemiscale_hpc.slurm.manager._whoami", return_value="alice"):
        yield


@pytest.fixture
def denver_tz(monkeypatch):
    """Run in a non-UTC local zone, so local/UTC mix-ups fail even on UTC CI."""
    monkeypatch.setenv("TZ", "America/Denver")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _local(ts: datetime) -> str:
    """Format ``ts`` as SLURM prints times with SLURM_TIME_FORMAT=standard."""
    return ts.astimezone().replace(tzinfo=None).isoformat(timespec="seconds")


def test_recover_jobs_tracks_own_queued_jobs(
    slurm_settings: SlurmManagerSettings, whoami, denver_tz
):
    """Own pending and running jobs are tracked with SLURM's submit times;
    other jobs, including ones of a manager whose name extends ours, are not."""
    name = slurm_settings.name
    pending, running = make_job_name(name), make_job_name(name)
    submitted = datetime.now(tz=timezone.utc).replace(microsecond=0) - timedelta(
        hours=3
    )
    squeue = (
        f"101|PENDING|{_local(submitted)}|{pending}\n"
        f"102|RUNNING|{_local(submitted)}|{running}\n"
        # Another manager whose name extends ours with a '.'.
        f"103|RUNNING|{_local(submitted)}|{name}.bar.{uuid4().hex}\n"
        f"104|RUNNING|{_local(submitted)}|other.{uuid4().hex}\n"
        f"105|RUNNING|{_local(submitted)}|interactive\n"
    )
    run, calls = _fake_slurm(squeue_user=squeue)
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run", side_effect=run):
        recovered = api.recover_jobs()

    assert recovered == [
        {"job_id": "101", "state": "PENDING", "name": pending},
        {"job_id": "102", "state": "RUNNING", "name": running},
    ]
    assert api.tracked_jobs == {"101", "102"}
    assert api._submission_times["101"] == submitted
    assert api._submission_times["102"] == submitted

    # Only squeue, scoped to the user and including the submit time.
    [cmd] = calls
    assert cmd[:3] == ["squeue", "-u", "alice"]
    # Plain ``squeue -u`` omits jobs on hidden partitions.
    assert "--all" in cmd
    assert "%i|%T|%V|%j" in cmd


def test_recover_jobs_ignores_slurm_environment_defaults(
    slurm_settings: SlurmManagerSettings, whoami
):
    """SQUEUE_*/SACCT_* defaults (e.g. SQUEUE_STATES=all, which would add
    finished jobs) are dropped, and times are requested as ISO 8601 whatever
    the user's SLURM_TIME_FORMAT is. Other variables pass through."""
    api = SlurmBatchApi(slurm_settings)
    user_env = {
        "SLURM_TIME_FORMAT": "relative",
        "SQUEUE_STATES": "all",
        "SQUEUE_PARTITION": "debug",
        "SACCT_FEDERATION": "1",
        "SLURM_CONF": "/etc/slurm/slurm.conf",
    }
    with patch.dict("os.environ", user_env), patch(
        "subprocess.run", return_value=_Result()
    ) as mock_run:
        api.recover_jobs(include_finished=True)
    assert mock_run.call_count == 2  # squeue and sacct
    for call in mock_run.call_args_list:
        env = call.kwargs["env"]
        assert env["SLURM_TIME_FORMAT"] == "standard"
        assert not any(k.startswith(("SQUEUE_", "SACCT_")) for k in env)
        assert env["SLURM_CONF"] == "/etc/slurm/slurm.conf"


def test_recover_jobs_unparseable_submit_time_grants_grace(
    slurm_settings: SlurmManagerSettings, whoami
):
    """A job whose submit time cannot be parsed is stamped with the current
    time, so it gets the full grace period rather than none."""
    job_name = make_job_name(slurm_settings.name)
    run, _ = _fake_slurm(squeue_user=f"101|RUNNING|Unknown|{job_name}\n")
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run", side_effect=run):
        api.recover_jobs()
    assert api.tracked_jobs == {"101"}
    assert api._job_age("101") < timedelta(seconds=60)


def test_recover_jobs_handles_empty_output(
    slurm_settings: SlurmManagerSettings, whoami
):
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run", return_value=_Result("\n")):
        assert api.recover_jobs() == []
    assert api.tracked_jobs == set()


def test_recover_jobs_skips_malformed_lines(
    slurm_settings: SlurmManagerSettings, whoami
):
    job_name = make_job_name(slurm_settings.name)
    run, _ = _fake_slurm(squeue_user=f"garbage\n101|PENDING|{job_name}\n")
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run", side_effect=run):
        assert api.recover_jobs() == []


def test_recover_jobs_raises_on_query_failure(
    slurm_settings: SlurmManagerSettings, whoami
):
    """A failed query raises rather than leaving tracked_jobs empty, which
    would let the manager over-submit."""
    api = SlurmBatchApi(slurm_settings)
    failure = _Result(stderr="slurm_load_jobs error: timeout\n", returncode=1)
    with patch("subprocess.run", return_value=failure) as mock_run, patch(
        "alchemiscale_hpc.slurm.manager.time.sleep"
    ) as sleep:
        with pytest.raises(
            RuntimeError, match="squeue exited with code 1 after 3 attempts.*timeout"
        ):
            api.recover_jobs()
    assert mock_run.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [2.0, 4.0]


def test_recover_jobs_retries_transient_failure(
    slurm_settings: SlurmManagerSettings, whoami
):
    """A query that fails once and then succeeds does not stop recovery."""
    job_name = make_job_name(slurm_settings.name)
    results = [
        _Result(stderr="Socket timed out\n", returncode=1),
        _Result(f"101|PENDING|Unknown|{job_name}\n"),
    ]
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run", side_effect=results), patch(
        "alchemiscale_hpc.slurm.manager.time.sleep"
    ) as sleep:
        recovered = api.recover_jobs()
    assert [job["job_id"] for job in recovered] == ["101"]
    sleep.assert_called_once()


def test_recover_jobs_include_finished_queries_sacct(
    slurm_settings: SlurmManagerSettings, whoami
):
    """With include_finished, recently completed and failed jobs come from
    sacct over an explicit time window, since it is not scoped by job ID."""
    name = slurm_settings.name
    failed = make_job_name(name)
    sacct = (
        f"201|FAILED|2026-01-01T00:00:00|{failed}\n"
        f"202|CANCELLED by 1000|2026-01-01T00:00:00|{name}.bar.{uuid4().hex}\n"
    )
    run, calls = _fake_slurm(sacct_user=sacct)
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run", side_effect=run):
        recovered = api.recover_jobs(include_finished=True)

    assert recovered == [{"job_id": "201", "state": "FAILED", "name": failed}]
    assert api.tracked_jobs == {"201"}
    sacct_cmd = calls[1]
    assert sacct_cmd[:3] == ["sacct", "-u", "alice"]
    assert sacct_cmd[sacct_cmd.index("-S") + 1] == "now-7days"
    states = sacct_cmd[sacct_cmd.index("-s") + 1].split(",")
    assert {"COMPLETED", "FAILED", "TIMEOUT"} <= set(states)
    assert "JobID,State,Submit,JobName" in sacct_cmd
    assert {"-X", "-P"} <= set(sacct_cmd)


def test_jobs_submitted_before_restart_are_recovered(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, whoami, denver_tz
):
    """After a restart, recovered pending jobs block new submissions and
    recovered running jobs are checked for registration."""
    name = slurm_settings.name
    pending, running = make_job_name(name), make_job_name(name)
    long_ago = _local(datetime.now(tz=timezone.utc) - timedelta(hours=1))

    # Before the restart: the manager submitted two jobs.
    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    before = SlurmBatchApi(slurm_settings)
    for job_id in ("101", "102"):
        with patch(
            "subprocess.run",
            return_value=_Result(f"Submitted batch job {job_id}\n"),
        ):
            before.submit_job(script)

    # After the restart: a fresh API knows nothing until it recovers.
    run, _ = _fake_slurm(
        squeue_user=(
            f"101|PENDING|{long_ago}|{pending}\n"
            f"102|RUNNING|{long_ago}|{running}\n"
        ),
        tracked={
            "PENDING": f"101|PENDING|N/A|{pending}\n",
            "RUNNING": f"102|RUNNING|{long_ago}|{running}\n",
        },
    )
    after = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run", side_effect=run):
        assert after.jobs_pending() is False
        after.recover_jobs()
        assert after.tracked_jobs == before.tracked_jobs
        assert after.jobs_pending() is True
        assert after.find_unregistered_jobs({running}) == []
        [job] = after.find_unregistered_jobs(set())
        assert job["name"] == running


def test_recovered_job_grace_period_runs_from_start_time(
    slurm_settings: SlurmManagerSettings, whoami, denver_tz
):
    """A recovered job's grace period runs from SLURM's start time, not from
    recovery or submission: a job submitted an hour ago that started 30s ago
    is still in its grace period."""
    settings = slurm_settings.model_copy(update={"job_registration_grace_period": 600})
    job_name = make_job_name(settings.name)
    now = datetime.now(tz=timezone.utc)
    submitted = _local(now - timedelta(hours=1))
    for started_ago, unregistered in ((30, False), (900, True)):
        started = _local(now - timedelta(seconds=started_ago))
        run, _ = _fake_slurm(
            squeue_user=f"101|RUNNING|{submitted}|{job_name}\n",
            tracked={"RUNNING": f"101|RUNNING|{started}|{job_name}\n"},
        )
        api = SlurmBatchApi(settings)
        with patch("subprocess.run", side_effect=run):
            api.recover_jobs()
            assert bool(api.find_unregistered_jobs(set())) is unregistered


def test_parse_slurm_time(denver_tz):
    """SLURM prints local time; America/Denver is UTC-7 in November."""
    assert _parse_slurm_time("2026-11-18T09:30:00") == datetime(
        2026, 11, 18, 16, 30, tzinfo=timezone.utc
    )
    assert _parse_slurm_time("Unknown") is None
    assert _parse_slurm_time("N/A") is None


def test_manager_start_recovers_jobs_before_first_cycle(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Queued jobs are recovered first; recently finished jobs separately,
    then their failures are cleared, all before the main loop."""
    mgr = _build_manager(slurm_settings, tmp_path)
    order = []
    with patch.object(
        mgr.batch_api,
        "recover_jobs",
        side_effect=lambda **kwargs: order.append(("recover", kwargs)) or [],
    ), patch.object(
        mgr.batch_api,
        "clear_failed_jobs",
        side_effect=lambda: order.append("clear_failed") or {},
    ), patch(
        "alchemiscale.compute.manager.ComputeManager.start",
        side_effect=lambda **kwargs: order.append("start"),
    ):
        mgr.start(max_cycles=1)
    assert order == [
        ("recover", {}),
        ("recover", {"include_finished": True}),
        "clear_failed",
        "start",
    ]


def test_manager_start_logs_failures_without_counting_them(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, whoami, caplog
):
    """Jobs that failed while the manager was down are summarized at WARNING
    and listed with their reasons at INFO, untracked, and not counted toward
    max_consecutive_failed_cycles, even with a limit of 0."""
    settings = slurm_settings.model_copy(update={"max_consecutive_failed_cycles": 0})
    mgr = _build_manager(settings, tmp_path)
    name = settings.name
    ts = "2026-11-18T09:30:00"
    run, _ = _fake_slurm(
        squeue_user=f"5|RUNNING|{ts}|{make_job_name(name)}\n",
        sacct_user=(
            f"7|TIMEOUT|{ts}|{make_job_name(name)}\n"
            f"8|COMPLETED|{ts}|{make_job_name(name)}\n"
        ),
        sacct_tracked="7|TIMEOUT|0:0|None|x\n",
    )
    with patch("subprocess.run", side_effect=run), patch(
        "alchemiscale.compute.manager.ComputeManager.start"
    ) as start:
        with caplog.at_level("INFO", logger="AlchemiscaleComputeManager"):
            mgr.start()
    start.assert_called_once()
    assert "Recovered 1 queued job(s) from the batch system (1 RUNNING)" in (
        caplog.text
    )
    records = [(r.levelname, r.message) for r in caplog.records]
    assert (
        "WARNING",
        "1 job(s) of this manager failed recently, before startup; not counted "
        "toward max_consecutive_failed_cycles",
    ) in records
    assert ("INFO", "Batch job 7 failed before startup: TIMEOUT, exit code 0") in (
        records
    )
    assert mgr.batch_api.tracked_jobs == {"5", "8"}
    assert mgr.consecutive_failed_cycles == 0


def test_manager_starts_when_accounting_is_unavailable(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, whoami, caplog
):
    """Reporting earlier failures is best effort: if sacct fails (accounting
    disabled, slurmdbd down), the manager logs a warning and starts, with its
    queued jobs recovered from squeue."""
    mgr = _build_manager(slurm_settings, tmp_path)
    job_name = make_job_name(slurm_settings.name)

    def run(cmd, **kwargs):
        if cmd[0] == "sacct":
            return _Result(
                stderr="sacct: error: Slurm accounting storage is disabled\n",
                returncode=1,
            )
        return _Result(f"5|RUNNING|2026-11-18T09:30:00|{job_name}\n")

    with patch("subprocess.run", side_effect=run), patch(
        "alchemiscale_hpc.slurm.manager.time.sleep"
    ), patch("alchemiscale.compute.manager.ComputeManager.start") as start:
        with caplog.at_level("WARNING", logger="AlchemiscaleComputeManager"):
            mgr.start()
    start.assert_called_once()
    assert mgr.batch_api.tracked_jobs == {"5"}
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert "failures from before startup are not reported" in record.message
    assert "accounting storage is disabled" in record.message


def test_manager_start_fails_if_recovery_fails(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, caplog
):
    """Starting without knowing which jobs are ours would over-submit. The
    reason reaches the manager log, not just the traceback."""
    mgr = _build_manager(slurm_settings, tmp_path)
    with patch.object(
        mgr.batch_api, "recover_jobs", side_effect=RuntimeError("squeue down")
    ), patch("alchemiscale.compute.manager.ComputeManager.start") as start:
        with caplog.at_level("ERROR", logger="AlchemiscaleComputeManager"):
            with pytest.raises(RuntimeError, match="squeue down"):
                mgr.start()
    start.assert_not_called()
    assert "Failed to recover jobs from the batch system: squeue down" in caplog.text


# ----------------------------------------------------------------------
# handle_finished_jobs: log and untrack, stop only on repeated failures
# ----------------------------------------------------------------------


def test_settings_max_consecutive_failed_cycles(tmp_path: Path):
    """Default 5, null allowed (never stop), negative rejected."""
    settings = SlurmManagerSettings(**_settings_kwargs(tmp_path))
    assert settings.max_consecutive_failed_cycles == 5
    settings = SlurmManagerSettings(
        **_settings_kwargs(tmp_path, max_consecutive_failed_cycles=None)
    )
    assert settings.max_consecutive_failed_cycles is None
    with pytest.raises(ValueError):
        SlurmManagerSettings(
            **_settings_kwargs(tmp_path, max_consecutive_failed_cycles=-1)
        )


def test_settings_cleanup_failed_jobs_false_warns(tmp_path: Path, caplog):
    """``cleanup_failed_jobs: false`` no longer keeps failed jobs tracked, so
    setting it warns, and the manager repeats the warning in its log."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SlurmManagerSettings(**_settings_kwargs(tmp_path))
        SlurmManagerSettings(**_settings_kwargs(tmp_path, cleanup_failed_jobs=True))
    with pytest.warns(FutureWarning, match="cleanup_failed_jobs=False is ignored"):
        settings = SlurmManagerSettings(
            **_settings_kwargs(tmp_path, cleanup_failed_jobs=False)
        )
    with caplog.at_level("WARNING", logger="AlchemiscaleComputeManager"):
        _build_manager(settings, tmp_path)
    assert "cleanup_failed_jobs=False is ignored" in caplog.text


def test_single_failed_job_is_logged_untracked_and_tolerated(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, caplog
):
    """One failed job is logged at ERROR with its reason and untracked; the
    manager carries on and submits."""
    mgr = _build_manager(slurm_settings, tmp_path)
    _track(mgr.batch_api, "7")
    _track(mgr.batch_api, "8")
    run, _ = _fake_slurm(sacct_tracked="7|NODE_FAIL|0:0|None|x\n")
    with patch("subprocess.run", side_effect=run), patch.object(
        mgr.batch_api, "submit_job", return_value="9"
    ) as submit:
        data = {"compute_service_ids": [], "num_tasks": 1}
        with caplog.at_level("ERROR", logger="AlchemiscaleComputeManager"):
            mgr.maintain_compute_services(data)
            n = mgr.create_compute_services(data, target=1)
    assert n == 1
    submit.assert_called_once()
    [record] = [r for r in caplog.records if "failed" in r.message]
    assert record.levelname == "ERROR"
    assert record.message == "Batch job 7 failed: NODE_FAIL, exit code 0"
    assert mgr.batch_api.tracked_jobs == {"8"}
    assert mgr.consecutive_failed_cycles == 1


def _check(mgr, failed=None, completed=(), registered=False):
    """Run handle_finished_jobs once: clear_failed_jobs returns ``failed``,
    clear_successful_jobs ``completed``, and with ``registered`` one of the
    manager's services is registered with the server."""
    server_job_names = (
        {make_job_name(mgr.settings.name)} if registered else {"other.job"}
    )
    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value=dict(failed or {})
    ), patch.object(
        mgr.batch_api, "clear_successful_jobs", return_value=list(completed)
    ):
        return mgr.handle_finished_jobs(server_job_names)


def test_failed_counter_resets_only_on_sign_of_health(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Checks with failures increment; checks with neither failures nor a
    sign of health (e.g. only pending jobs) leave the count; a completed job
    or a registered service resets it."""
    mgr = _build_manager(slurm_settings, tmp_path)
    for i in range(3):
        _check(mgr, {str(i): "FAILED"})
    assert mgr.consecutive_failed_cycles == 3
    _check(mgr)
    assert mgr.consecutive_failed_cycles == 3
    _check(mgr, completed=["9"])
    assert mgr.consecutive_failed_cycles == 0
    _check(mgr, {"4": "FAILED"})
    _check(mgr, registered=True)
    assert mgr.consecutive_failed_cycles == 0


def test_other_managers_services_are_not_a_sign_of_health(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Only this manager's registered services count: not those of manager
    ``testmgr.bar``, whose job names extend ours."""
    mgr = _build_manager(slurm_settings, tmp_path)
    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={"1": "FAILED"}
    ), patch.object(mgr.batch_api, "clear_successful_jobs", return_value=[]):
        mgr.handle_finished_jobs({make_job_name(f"{mgr.settings.name}.bar")})
    assert mgr.consecutive_failed_cycles == 1


def test_healthy_fleet_with_sporadic_failures_never_stops(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """In a large fleet, some job is preempted or times out on most checks;
    with services registered, that never stops the manager."""
    mgr = _build_manager(slurm_settings, tmp_path)
    for i in range(200):
        _check(mgr, {str(i): "PREEMPTED"}, registered=True)
    assert mgr.consecutive_failed_cycles == 0


def _simulate_broken_template(mgr, pend, cycles=100):
    """Every job pends ``pend`` cycles, then fails without registering.
    Returns the cycle at which the manager stopped, or None."""
    jobs = {}  # job ID -> cycle submitted
    state = {"cycle": 0}

    def failed(jid):
        return state["cycle"] - jobs[jid] > pend

    def clear_failed():
        out = {jid: "FAILED, exit code 1" for jid in jobs if failed(jid)}
        for jid in out:
            del jobs[jid]
        return out

    def submit(path):
        jid = str(len(jobs) + 1000 * state["cycle"])
        jobs[jid] = state["cycle"]
        return jid

    api = mgr.batch_api
    with patch.object(api, "clear_failed_jobs", side_effect=clear_failed), patch.object(
        api, "clear_successful_jobs", return_value=[]
    ), patch.object(api, "find_unregistered_jobs", return_value=[]), patch.object(
        api, "jobs_pending", side_effect=lambda: any(not failed(j) for j in jobs)
    ), patch.object(
        api, "submit_job", side_effect=submit
    ):
        for cycle in range(cycles):
            state["cycle"] = cycle
            data = {"compute_service_ids": [], "num_tasks": 10}
            try:
                mgr.maintain_compute_services(data)
                mgr.create_compute_services(data, target=2)
            except JobFailureError:
                return cycle
    return None


@pytest.mark.parametrize("pend", [0, 1, 2, 5])
def test_broken_template_stops_even_when_jobs_pend(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, pend: int
):
    """Checks that only see pending jobs do not reset the count, so jobs that
    pend before failing still stop the manager, on the 6th failing check."""
    mgr = _build_manager(slurm_settings, tmp_path)
    assert _simulate_broken_template(mgr, pend) == 6 * (pend + 1)


def test_failed_cycles_raise_after_default_limit(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """With the default limit of 5, the 6th failing check without a sign of
    health raises, naming the jobs and pointing at the template and
    settings."""
    mgr = _build_manager(slurm_settings, tmp_path)
    for i in range(5):
        _check(mgr, {str(i): "FAILED"})
    with pytest.raises(
        JobFailureError,
        match=r"6 cycles with no job completing or registering in between, "
        r"exceeding max_consecutive_failed_cycles=5; most recent: 5, 6\. "
        r"Check the job script template",
    ):
        _check(mgr, {"5": "FAILED", "6": "TIMEOUT"})


def test_failed_cycles_limit_zero_stops_on_first_failure(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    mgr = _build_manager(
        slurm_settings.model_copy(update={"max_consecutive_failed_cycles": 0}),
        tmp_path,
    )
    _check(mgr)
    with pytest.raises(JobFailureError):
        _check(mgr, {"1": "FAILED"})


def test_failed_cycles_limit_none_never_stops(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    mgr = _build_manager(
        slurm_settings.model_copy(update={"max_consecutive_failed_cycles": None}),
        tmp_path,
    )
    for i in range(50):
        _check(mgr, {str(i): "FAILED"})
    assert mgr.consecutive_failed_cycles == 50


# ----------------------------------------------------------------------
# handle_unregistered_jobs: tolerate briefly unregistered running jobs
# ----------------------------------------------------------------------


def _check_unregistered(mgr, job_name, checks):
    """Run handle_unregistered_jobs ``checks`` times with job 1 unregistered."""
    job = {"job_id": "1", "name": job_name, "state": "RUNNING"}
    with patch.object(mgr.batch_api, "find_unregistered_jobs", return_value=[job]):
        for _ in range(checks):
            mgr.handle_unregistered_jobs(set())


def test_unregistered_job_is_tolerated_then_raises(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, caplog
):
    """With grace 0, the first check warns and the second raises, as with
    alchemiscale-k8s's watchlist."""
    mgr = _build_manager(slurm_settings, tmp_path)
    job_name = make_job_name(slurm_settings.name)
    with caplog.at_level("WARNING", logger="AlchemiscaleComputeManager"):
        _check_unregistered(mgr, job_name, 1)
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert f"Job {job_name} (batch ID: 1) is running but not registered" in (
        record.message
    )
    with pytest.raises(
        JobNotFoundError,
        match=re.escape(job_name) + r".*2 consecutive checks.*--name",
    ):
        _check_unregistered(mgr, job_name, 1)


def test_unregistered_job_must_stay_unregistered_for_grace_period(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """With short cycles, a job is tolerated until it has been found
    unregistered for at least the grace period, covering e.g. a slow epilog
    after its service deregistered."""
    mgr = _build_manager(
        slurm_settings.model_copy(update={"job_registration_grace_period": 120}),
        tmp_path,
    )
    job_name = make_job_name(slurm_settings.name)
    _check_unregistered(mgr, job_name, 5)
    # Back-date when the job was first found unregistered.
    first_found, checks = mgr._unregistered_jobs["1"]
    mgr._unregistered_jobs["1"] = (first_found - timedelta(seconds=121), checks)
    with pytest.raises(JobNotFoundError, match="6 consecutive checks"):
        _check_unregistered(mgr, job_name, 1)


def test_unregistered_job_is_forgotten_once_registered_or_gone(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """A check that does not find the job unregistered forgets it."""
    mgr = _build_manager(slurm_settings, tmp_path)
    job_name = make_job_name(slurm_settings.name)
    _check_unregistered(mgr, job_name, 1)
    with patch.object(mgr.batch_api, "find_unregistered_jobs", return_value=[]):
        mgr.handle_unregistered_jobs({job_name})
    assert mgr._unregistered_jobs == {}
    _check_unregistered(mgr, job_name, 1)  # a first check again: no raise


def test_unregistered_running_job_end_to_end(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, denver_tz
):
    """Through squeue: a job that pended for an hour and just started is in
    its grace period; once past it, it is tolerated once, then raises."""
    settings = slurm_settings.model_copy(update={"job_registration_grace_period": 120})
    mgr = _build_manager(settings, tmp_path)
    job_name = make_job_name(settings.name)
    _track(mgr.batch_api, "1", age_seconds=3600)
    now = datetime.now(tz=timezone.utc)

    def check(started_ago):
        started = _local(now - timedelta(seconds=started_ago))
        run, _ = _fake_slurm(tracked={"RUNNING": f"1|RUNNING|{started}|{job_name}\n"})
        with patch("subprocess.run", side_effect=run):
            mgr.handle_unregistered_jobs(set())

    check(10)
    assert mgr._unregistered_jobs == {}
    check(200)
    assert mgr._unregistered_jobs["1"][1] == 1
    # The next check comes one (150s) cycle later.
    mgr._unregistered_jobs["1"] = (now - timedelta(seconds=150), 1)
    with pytest.raises(JobNotFoundError):
        check(350)


# ----------------------------------------------------------------------
# Submission failures
# ----------------------------------------------------------------------


def _submit_cycles(mgr, outcomes, pending=False, registered=False):
    """Run one create_compute_services cycle per entry of ``outcomes``, each
    a list of submit_job results (an exception or a job ID), one per job the
    cycle tries to submit. With ``registered``, one of the manager's
    services is registered with the server."""
    results = [r for outcome in outcomes for r in outcome]
    service_ids = (
        [f"{make_job_name(mgr.settings.name)}-{uuid4().hex}"] if registered else []
    )
    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={}
    ), patch.object(
        mgr.batch_api, "find_unregistered_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "clear_successful_jobs", return_value=[]
    ), patch.object(
        mgr.batch_api, "jobs_pending", return_value=pending
    ), patch.object(
        mgr.batch_api, "submit_job", side_effect=results
    ):
        for outcome in outcomes:
            mgr.create_compute_services(
                {"compute_service_ids": service_ids, "num_tasks": 10},
                target=max(len(outcome), 1),
            )


def test_submission_failures_counted_and_reset(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Cycles in which every submission fails are counted; a cycle with any
    successful submission resets the count."""
    mgr = _build_manager(slurm_settings, tmp_path)
    fail = RuntimeError("sbatch: error: invalid partition specified")
    _submit_cycles(mgr, [[fail], [fail]])
    assert mgr.consecutive_failed_submission_cycles == 2
    # The first submission succeeds and the second fails: a reset.
    _submit_cycles(mgr, [["1", fail]])
    assert mgr.consecutive_failed_submission_cycles == 0


def test_submission_failures_raise_after_limit(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    mgr = _build_manager(slurm_settings, tmp_path)
    fail = RuntimeError("sbatch: error: invalid partition specified")
    _submit_cycles(mgr, [[fail]] * 5)
    with pytest.raises(
        JobSubmissionError,
        match=r"failed in 6 cycles with no compute service of this manager "
        r"registered, exceeding max_consecutive_failed_cycles=5; most recent "
        r"error: sbatch: error: invalid partition specified\. Check the job "
        r"script template",
    ) as excinfo:
        _submit_cycles(mgr, [[fail]])
    assert excinfo.value.__cause__ is fail


def test_submission_failures_limit_zero_and_none(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    fail = RuntimeError("boom")
    mgr = _build_manager(
        slurm_settings.model_copy(update={"max_consecutive_failed_cycles": 0}),
        tmp_path,
    )
    with pytest.raises(JobSubmissionError):
        _submit_cycles(mgr, [[fail]])

    mgr = _build_manager(
        slurm_settings.model_copy(update={"max_consecutive_failed_cycles": None}),
        tmp_path,
    )
    _submit_cycles(mgr, [[fail]] * 20)
    assert mgr.consecutive_failed_submission_cycles == 20


def test_submission_failures_not_counted_while_service_registered(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """At a submit limit (e.g. MaxSubmitJobsPerUser), every sbatch fails
    while the fleet runs; that does not stop a healthy manager."""
    mgr = _build_manager(slurm_settings, tmp_path)
    limit = RuntimeError(
        "sbatch exited with code 1: sbatch: error: QOSMaxSubmitJobPerUserLimit"
    )
    _submit_cycles(mgr, [[limit]])
    _submit_cycles(mgr, [[limit]] * 50, registered=True)
    assert mgr.consecutive_failed_submission_cycles == 1


def test_cycle_without_submission_leaves_submission_count(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """A cycle that skips submission because jobs are pending neither counts
    as a failure nor resets the count."""
    mgr = _build_manager(slurm_settings, tmp_path)
    _submit_cycles(mgr, [[RuntimeError("boom")]])
    _submit_cycles(mgr, [[]], pending=True)
    assert mgr.consecutive_failed_submission_cycles == 1


def test_job_script_write_failure_counts_as_failed_submission(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Failing to write the job script (e.g. an unwritable job_script_dir) is
    a failed submission, not an immediate halt."""
    mgr = _build_manager(slurm_settings, tmp_path)
    with patch.object(
        mgr, "_create_job_script", side_effect=PermissionError("Permission denied")
    ):
        _submit_cycles(mgr, [[]])
    assert mgr.consecutive_failed_submission_cycles == 1


def test_job_script_rendering_error_propagates(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Other errors while creating the script are bugs (e.g. a template
    placeholder mismatch), so they propagate rather than being counted."""
    mgr = _build_manager(slurm_settings, tmp_path)
    with patch.object(
        mgr, "_create_job_script", side_effect=ValueError("unknown placeholder")
    ):
        with pytest.raises(ValueError, match="unknown placeholder"):
            _submit_cycles(mgr, [[]])
    assert mgr.consecutive_failed_submission_cycles == 0


def test_submit_job_error_includes_sbatch_stderr(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """sbatch's stderr says why it failed; it reaches the error message."""
    api = SlurmBatchApi(slurm_settings)
    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    error = subprocess.CalledProcessError(
        1,
        ["sbatch", str(script)],
        output="",
        stderr="sbatch: error: invalid partition specified: nope\n",
    )
    with patch("subprocess.run", side_effect=error):
        with pytest.raises(
            RuntimeError,
            match=r"^sbatch exited with code 1: sbatch: error: invalid "
            r"partition specified: nope$",
        ) as excinfo:
            api.submit_job(script)
    assert excinfo.value.__cause__ is error
    assert api.tracked_jobs == set()


# Job script template validation
# ----------------------------------------------------------------------

EXAMPLE_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "examples" / "slurm" / "job-template.sh"
)


@pytest.mark.parametrize("missing", ["JOB_NAME", "COMPUTE_MANAGER_ID"])
def test_template_missing_required_placeholder_raises(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, missing: str
):
    """A template without a required placeholder is rejected at construction."""
    template = slurm_settings.job_script_template
    template.write_text(template.read_text().replace(f"{{{{ {missing} }}}}", "x"))

    with pytest.raises(ValueError, match=f"missing required placeholder.*{missing}"):
        _build_manager(slurm_settings, tmp_path)


def test_template_unknown_placeholder_warns(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, caplog
):
    """Unknown or mis-spaced ``{{ ... }}`` tokens are logged as warnings."""
    template = slurm_settings.job_script_template
    template.write_text(
        template.read_text()
        + "{{ PARITION }}\necho {{JOB_NAME}}\necho {{ JOB_NAME\n}}\n"
    )

    with caplog.at_level("WARNING"):
        _build_manager(slurm_settings, tmp_path)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 3
    assert any("{{ PARITION }}" in w for w in warnings)
    assert any("{{JOB_NAME}}" in w for w in warnings)
    assert any("{{ JOB_NAME\n}}" in w for w in warnings)


@pytest.mark.parametrize(
    "token", ["{{ PARTITION }}", "{{ACCOUNT}}", "#SBATCH --qos={{ QOS }}"]
)
def test_template_removed_placeholder_raises(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, token: str
):
    """Templates using the old PARTITION/ACCOUNT/QOS placeholders are rejected."""
    template = slurm_settings.job_script_template
    template.write_text(template.read_text() + token + "\n")

    with pytest.raises(ValueError, match="no longer substituted"):
        _build_manager(slurm_settings, tmp_path)


def test_example_template_validates_cleanly(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, caplog
):
    """The bundled example template passes validation without warnings and
    renders with no placeholders left over."""
    settings = slurm_settings.model_copy(
        update={"job_script_template": EXAMPLE_TEMPLATE}
    )

    with caplog.at_level("WARNING"):
        mgr = _build_manager(settings, tmp_path)

    assert not [r for r in caplog.records if r.levelname == "WARNING"]

    script_path = mgr._create_job_script()
    try:
        assert "{{" not in script_path.read_text()
    finally:
        script_path.unlink(missing_ok=True)


def test_render_template_rejects_undeclared_substitutions(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Substitution keys must match ``TEMPLATE_PLACEHOLDERS`` exactly."""
    mgr = _build_manager(slurm_settings, tmp_path)
    substitutions = dict.fromkeys(SlurmManager.TEMPLATE_PLACEHOLDERS, "")

    with pytest.raises(ValueError, match="TEMPLATE_PLACEHOLDERS"):
        mgr._render_template({**substitutions, "EXTRA": ""})
    with pytest.raises(ValueError, match="TEMPLATE_PLACEHOLDERS"):
        mgr._render_template({k: v for k, v in substitutions.items() if k != "JOB_NAME"})
