"""Tests for the SLURM batch API and manager."""

import re
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from alchemiscale.storage.models import ComputeManagerID, ComputeServiceID

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
        "{{ PARTITION }}\n{{ ACCOUNT }}\n{{ QOS }}\n"
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
    template.write_text("#!/bin/bash\n")
    kwargs = dict(
        name="testmgr",
        logfile=None,
        max_compute_services=1,
        job_script_template=template,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "name", ["testmgr", "mycluster.gpu", "my_cluster", "Site:queue.v2", "a"]
)
def test_settings_accepts_valid_name(tmp_path: Path, name: str):
    settings = SlurmManagerSettings(**_settings_kwargs(tmp_path, name=name))
    assert settings.name == name
    # Catch drift from upstream's ID rules: names we accept must yield IDs
    # alchemiscale accepts, for both the manager and its jobs' services.
    ComputeManagerID.new_from_name(name)
    ComputeServiceID.new_from_name(make_job_name(name))


@pytest.mark.parametrize(
    "name",
    ["my-slurm-manager", "1cluster", "_cluster", ".cluster", "my cluster", ""],
)
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


def test_generate_job_name_identifies_manager(tmp_path: Path):
    """Managers with different names produce distinguishable job names."""
    prefixes = set()
    for mgr_name in ("mgra", "mgrb"):
        settings = SlurmManagerSettings(**_settings_kwargs(tmp_path, name=mgr_name))
        mgr = _build_manager(settings, tmp_path)
        prefixes.add(mgr._generate_job_name().rsplit(".", 1)[0])
    assert prefixes == {"mgra", "mgrb"}


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
            "NODE_FAIL, exit code 0, reason ReqNodeNotAvail",
        ),
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
    assert "CANCELLED" in cmd[cmd.index("-s") + 1].split(",")


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
    with patch("alchemiscale.compute.manager.AlchemiscaleComputeManagerClient"):
        return SlurmManager(
            settings=slurm_settings, service_settings_path=service_config
        )


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
        assert "#SBATCH --partition=testpart" in content
        assert "#SBATCH --account=testacct" in content
        # qos was unset -> empty substitution
        assert "{{ QOS }}" not in content
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
    ), patch.object(mgr.batch_api, "clear_successful_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", side_effect=_fake_submit
    ):
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
    ), patch.object(mgr.batch_api, "clear_successful_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", side_effect=_fake_submit
    ):
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
    ), patch.object(mgr.batch_api, "clear_successful_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", side_effect=_fake_submit
    ):
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
    ), patch.object(mgr.batch_api, "clear_successful_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=True
    ), patch.object(
        mgr.batch_api, "submit_job"
    ) as submit:
        n = mgr.create_compute_services(
            {"compute_service_ids": [], "num_tasks": 5}, target=5
        )
    assert n == 0
    submit.assert_not_called()


def test_create_compute_services_strips_uuid_suffix_correctly(
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
    ), patch.object(mgr.batch_api, "clear_successful_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=True
    ):
        mgr.create_compute_services(
            {"compute_service_ids": [service_id], "num_tasks": 0}, target=0
        )

    assert captured["names"] == {service_name}


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
):
    """Fake ``subprocess.run`` for squeue/sacct.

    ``squeue_user``/``sacct_user`` answer user-scoped queries (recovery);
    ``tracked`` maps a ``--states=`` value to the answer for ``--jobs=``-scoped
    squeue queries, and ``sacct_tracked`` answers ``--jobs=``-scoped sacct
    queries (the manager loop). Every call is recorded in ``calls``.
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
    assert order == [("recover", {"include_finished": True}), "clear_failed", "start"]


def test_manager_start_logs_failures_without_counting_them(
    slurm_settings: SlurmManagerSettings, tmp_path: Path, whoami, caplog
):
    """Jobs that failed while the manager was down are logged with their
    reasons at startup, untracked, and not counted toward
    max_consecutive_failed_cycles, even with a limit of 0."""
    settings = slurm_settings.model_copy(
        update={"max_consecutive_failed_cycles": 0}
    )
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
    assert "Recovered 3 job(s) from the batch system" in caplog.text
    [record] = [r for r in caplog.records if "failed before startup" in r.message]
    assert record.levelname == "WARNING"
    assert record.message == "Batch job 7 failed before startup: TIMEOUT, exit code 0"
    assert mgr.batch_api.tracked_jobs == {"5", "8"}
    assert mgr.consecutive_failed_cycles == 0


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
# handle_failed_jobs: log and untrack, stop only on repeated failures
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
        with caplog.at_level("ERROR", logger="AlchemiscaleComputeManager"):
            n = mgr.create_compute_services(
                {"compute_service_ids": [], "num_tasks": 1}, target=1
            )
    assert n == 1
    submit.assert_called_once()
    [record] = [r for r in caplog.records if "failed" in r.message]
    assert record.levelname == "ERROR"
    assert record.message == "Batch job 7 failed: NODE_FAIL, exit code 0"
    assert mgr.batch_api.tracked_jobs == {"8"}
    assert mgr.consecutive_failed_cycles == 1


def _check_failures(mgr, failures):
    """Run handle_failed_jobs once per entry of ``failures``, each the
    ``{job_id: reason}`` that clear_failed_jobs returns on that check."""
    with patch.object(mgr.batch_api, "clear_failed_jobs", side_effect=failures):
        for _ in failures:
            mgr.handle_failed_jobs()


def test_failed_cycle_counter_increments_and_resets(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    mgr = _build_manager(slurm_settings, tmp_path)
    _check_failures(mgr, [{"1": "FAILED"}, {"2": "FAILED"}, {"3": "TIMEOUT"}])
    assert mgr.consecutive_failed_cycles == 3
    _check_failures(mgr, [{}])
    assert mgr.consecutive_failed_cycles == 0
    _check_failures(mgr, [{"4": "FAILED"}])
    assert mgr.consecutive_failed_cycles == 1


def test_failed_cycles_raise_after_default_limit(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """With the default limit of 5, the 6th consecutive failing check raises,
    naming the jobs and pointing at the template and settings."""
    mgr = _build_manager(slurm_settings, tmp_path)
    _check_failures(mgr, [{str(i): "FAILED"} for i in range(5)])
    with pytest.raises(
        JobFailureError,
        match=r"6 consecutive cycles, exceeding max_consecutive_failed_cycles=5; "
        r"most recent: 5, 6\. Check the job script template",
    ):
        _check_failures(mgr, [{"5": "FAILED", "6": "TIMEOUT"}])


def test_failed_cycles_limit_zero_stops_on_first_failure(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    mgr = _build_manager(
        slurm_settings.model_copy(update={"max_consecutive_failed_cycles": 0}),
        tmp_path,
    )
    _check_failures(mgr, [{}])
    with pytest.raises(JobFailureError):
        _check_failures(mgr, [{"1": "FAILED"}])


def test_failed_cycles_limit_none_never_stops(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    mgr = _build_manager(
        slurm_settings.model_copy(update={"max_consecutive_failed_cycles": None}),
        tmp_path,
    )
    _check_failures(mgr, [{str(i): "FAILED"} for i in range(50)])
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


def _submit_cycles(mgr, outcomes, pending=False):
    """Run one create_compute_services cycle per entry of ``outcomes``, each
    a list of submit_job results (an exception or a job ID), one per job the
    cycle tries to submit."""
    results = [r for outcome in outcomes for r in outcome]
    with patch.object(
        mgr.batch_api, "clear_failed_jobs", return_value={}
    ), patch.object(
        mgr.batch_api, "find_unregistered_jobs", return_value=[]
    ), patch.object(mgr.batch_api, "clear_successful_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=pending
    ), patch.object(
        mgr.batch_api, "submit_job", side_effect=results
    ):
        for outcome in outcomes:
            mgr.create_compute_services(
                {"compute_service_ids": [], "num_tasks": 10},
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
        match=r"failed in 6 consecutive cycles, exceeding "
        r"max_consecutive_failed_cycles=5; most recent error: sbatch: error: "
        r"invalid partition specified\. Check the job script template",
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


def test_cycle_without_submission_leaves_submission_count(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """A cycle that skips submission because jobs are pending neither counts
    as a failure nor resets the count."""
    mgr = _build_manager(slurm_settings, tmp_path)
    _submit_cycles(mgr, [[RuntimeError("boom")]])
    _submit_cycles(mgr, [[]], pending=True)
    assert mgr.consecutive_failed_submission_cycles == 1


def test_job_script_failure_counts_as_failed_submission(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """Failing to write the job script (e.g. an unwritable job_script_dir) is
    a failed submission, not an immediate halt."""
    mgr = _build_manager(slurm_settings, tmp_path)
    with patch.object(
        mgr, "_create_job_script", side_effect=OSError("Permission denied")
    ):
        _submit_cycles(mgr, [[]])
    assert mgr.consecutive_failed_submission_cycles == 1
