"""Tests for the SLURM batch API and manager."""

import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from alchemiscale.storage.models import ComputeManagerID, ComputeServiceID
from alchemiscale.storage.models import ComputeManagerInstruction

from alchemiscale_hpc import (
    HPCBatchApi,
    JobFailureError,
    JobNotFoundError,
)
from alchemiscale_hpc.base import is_own_job_name, make_job_name
from alchemiscale_hpc.slurm import (
    SlurmBatchApi,
    SlurmManager,
    SlurmManagerSettings,
)


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
# check_job_health
# ----------------------------------------------------------------------


def test_check_job_health_raises_on_failure(slurm_settings: SlurmManagerSettings):
    """``_get_failed_jobs`` is already scoped to tracked jobs, so any
    returned entry is a real failure to surface."""
    api = SlurmBatchApi(slurm_settings)
    _track(api, "12345")
    with patch.object(
        api,
        "_get_failed_jobs",
        return_value=[{"job_id": "12345", "name": "x", "state": "FAILED"}],
    ):
        with pytest.raises(JobFailureError, match="12345"):
            api.check_job_health()


def test_check_job_health_silent_when_no_failures(
    slurm_settings: SlurmManagerSettings,
):
    api = SlurmBatchApi(slurm_settings)
    _track(api, "12345")
    with patch.object(api, "_get_failed_jobs", return_value=[]):
        api.check_job_health()


# ----------------------------------------------------------------------
# verify_running_jobs
# ----------------------------------------------------------------------


def test_verify_running_jobs_recognizes_registered(
    slurm_settings: SlurmManagerSettings,
):
    """A tracked SLURM job whose name matches a server-known service passes."""
    api = SlurmBatchApi(slurm_settings)
    _track(api, "1")
    job_name = f"{slurm_settings.name}.deadbeef"
    with patch.object(
        api,
        "_get_running_jobs",
        return_value=[{"job_id": "1", "name": job_name, "state": "RUNNING"}],
    ):
        api.verify_running_jobs({job_name})


def test_verify_running_jobs_raises_on_unregistered(
    slurm_settings: SlurmManagerSettings,
):
    """A tracked, past-grace, running job not in server_job_names triggers."""
    api = SlurmBatchApi(slurm_settings)
    _track(api, "1")
    job_name = f"{slurm_settings.name}.deadbeef"
    with patch.object(
        api,
        "_get_running_jobs",
        return_value=[{"job_id": "1", "name": job_name, "state": "RUNNING"}],
    ):
        with pytest.raises(JobNotFoundError, match=re.escape(job_name)):
            api.verify_running_jobs(set())


def test_verify_running_jobs_ignores_untracked_jobs(
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
        return_value=[
            {
                "job_id": "1",
                "name": f"{slurm_settings.name}.fromothermanager",
                "state": "RUNNING",
            }
        ],
    ):
        api.verify_running_jobs(set())  # should NOT raise


def test_verify_running_jobs_grace_period_skips_young_jobs(
    slurm_settings: SlurmManagerSettings,
):
    """Jobs younger than the grace period are skipped, even when unregistered."""
    settings = slurm_settings.model_copy(update={"job_registration_grace_period": 600})
    api = SlurmBatchApi(settings)
    # Track with age=10s; well inside the 600s grace window.
    _track(api, "1", age_seconds=10)
    job_name = f"{settings.name}.young"
    with patch.object(
        api,
        "_get_running_jobs",
        return_value=[{"job_id": "1", "name": job_name, "state": "RUNNING"}],
    ):
        # Empty server_job_names; without the grace period this would raise.
        api.verify_running_jobs(set())


def test_verify_running_jobs_grace_period_expires(
    slurm_settings: SlurmManagerSettings,
):
    """Past the grace period, the same job DOES raise."""
    settings = slurm_settings.model_copy(update={"job_registration_grace_period": 60})
    api = SlurmBatchApi(settings)
    _track(api, "1", age_seconds=120)
    job_name = f"{settings.name}.old"
    with patch.object(
        api,
        "_get_running_jobs",
        return_value=[{"job_id": "1", "name": job_name, "state": "RUNNING"}],
    ):
        with pytest.raises(JobNotFoundError, match="grace period"):
            api.verify_running_jobs(set())


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
        return_value=[{"job_id": "1", "name": "x", "state": "FAILED"}],
    ):
        api.clear_failed_jobs()
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
        {"stdout": "100|RUNNING|x\n", "stderr": "", "returncode": 0},
    )
    with patch("subprocess.run", return_value=fake) as mock_run:
        api._squeue_tracked("RUNNING")
    call_args = mock_run.call_args[0][0]
    assert "--jobs=100,200" in call_args
    assert "--states=RUNNING" in call_args


def test_sacct_query_skips_when_no_tracked_jobs(
    slurm_settings: SlurmManagerSettings,
):
    """No tracked jobs -> no sacct call. Avoids scanning the user's history."""
    api = SlurmBatchApi(slurm_settings)
    with patch("subprocess.run") as mock_run:
        assert api._sacct_query("COMPLETED", set()) == []
    mock_run.assert_not_called()


def test_sacct_query_uses_jobs_filter_and_parsable_output(
    slurm_settings: SlurmManagerSettings,
):
    """sacct is called with ``--jobs=`` (sidesteps lookback) and ``-P``
    (pipe-separated parsable output)."""
    api = SlurmBatchApi(slurm_settings)
    fake = type("R", (), {"stdout": "", "stderr": "", "returncode": 0})
    with patch("subprocess.run", return_value=fake) as mock_run:
        api._sacct_query("COMPLETED", {"100", "200"})
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

    with patch.object(mgr.batch_api, "check_job_health"), patch.object(
        mgr.batch_api, "verify_running_jobs", side_effect=_capture
    ), patch.object(mgr.batch_api, "clear_successful_jobs"):
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
        mgr.batch_api, "check_job_health"
    ) as health, patch.object(
        mgr.batch_api, "verify_running_jobs"
    ) as verify, patch.object(
        mgr.batch_api, "clear_successful_jobs"
    ) as clear, patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", return_value="1"
    ):
        mgr.create_compute_services(
            {"compute_service_ids": [], "num_tasks": 1}, target=1
        )
    maintain.assert_not_called()
    health.assert_not_called()
    verify.assert_not_called()
    clear.assert_not_called()


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

    with patch.object(mgr.batch_api, "check_job_health") as health, patch.object(
        mgr.batch_api, "verify_running_jobs"
    ) as verify, patch.object(
        mgr.batch_api, "clear_successful_jobs"
    ) as clear, patch.object(
        mgr, "create_compute_services"
    ) as create:
        assert mgr.cycle() is True

    health.assert_called_once_with()
    verify.assert_called_once_with({service_name})
    clear.assert_called_once_with()
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

    with patch.object(mgr.batch_api, "check_job_health") as health, patch.object(
        mgr.batch_api, "verify_running_jobs"
    ) as verify, patch.object(
        mgr.batch_api, "clear_successful_jobs"
    ) as clear, patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", return_value="1"
    ) as submit:
        assert mgr.cycle() is True

    health.assert_called_once_with()
    verify.assert_called_once_with(set())
    clear.assert_called_once_with()
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
    """A failing health check aborts the cycle before sizing or submission,
    and the client's ``get_instruction`` is restored afterwards."""
    mgr = _build_manager(slurm_settings, tmp_path)
    get_instruction = mgr.client.get_instruction
    get_instruction.return_value = (
        ComputeManagerInstruction.OK,
        {"compute_service_ids": [], "num_tasks": 1},
    )

    with patch.object(
        mgr.batch_api, "check_job_health", side_effect=JobFailureError("boom")
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
