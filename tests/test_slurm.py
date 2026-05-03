"""Tests for the SLURM batch API and manager."""

from pathlib import Path
from unittest.mock import patch

import pytest

from alchemiscale_hpc import (
    HPCBatchApi,
    JobFailureError,
    JobNotFoundError,
)
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
    )


def test_slurm_batch_api_is_concrete(slurm_settings: SlurmManagerSettings):
    """SlurmBatchApi implements every abstract method on HPCBatchApi."""
    api = SlurmBatchApi(slurm_settings)
    assert isinstance(api, HPCBatchApi)
    # tracked_jobs initialized empty
    assert api.tracked_jobs == set()


def test_slurm_settings_extends_script_template_base():
    """SlurmManagerSettings exposes both base and script-template fields."""
    fields = set(SlurmManagerSettings.model_fields)
    # Inherited from HPCManagerSettings:
    assert {"job_name_prefix", "max_submit_per_cycle"} <= fields
    # Inherited from ScriptTemplateHPCManagerSettings:
    assert {
        "job_script_template",
        "untrack_completed_jobs",
        "untrack_failed_jobs",
        "keep_job_scripts",
        "job_script_dir",
    } <= fields
    # SLURM-specific:
    assert {
        "partition",
        "account",
        "qos",
        "submit_command",
        "cancel_command",
        "query_command",
        "accounting_command",
    } <= fields


def test_parse_csv_jobs_handles_empty():
    assert SlurmBatchApi._parse_csv_jobs("") == []
    assert SlurmBatchApi._parse_csv_jobs("\n\n") == []


def test_parse_csv_jobs_parses_squeue_output():
    out = "12345,alchemiscale-abcd,RUNNING\n67890,otherjob,PENDING\n"
    parsed = SlurmBatchApi._parse_csv_jobs(out)
    assert parsed == [
        {"job_id": "12345", "name": "alchemiscale-abcd", "state": "RUNNING"},
        {"job_id": "67890", "name": "otherjob", "state": "PENDING"},
    ]


def test_parse_whitespace_jobs_parses_sacct_output():
    out = "12345 myjob COMPLETED\n67890 other FAILED\n"
    parsed = SlurmBatchApi._parse_whitespace_jobs(out)
    assert parsed == [
        {"job_id": "12345", "name": "myjob", "state": "COMPLETED"},
        {"job_id": "67890", "name": "other", "state": "FAILED"},
    ]


def test_check_job_health_raises_on_tracked_failure(
    slurm_settings: SlurmManagerSettings,
):
    api = SlurmBatchApi(slurm_settings)
    api.tracked_jobs.add("12345")
    with patch.object(
        api,
        "_get_failed_jobs",
        return_value=[{"job_id": "12345", "name": "x", "state": "FAILED"}],
    ):
        with pytest.raises(JobFailureError, match="12345"):
            api.check_job_health()


def test_check_job_health_ignores_untracked_failures(
    slurm_settings: SlurmManagerSettings,
):
    api = SlurmBatchApi(slurm_settings)
    api.tracked_jobs.add("99999")
    with patch.object(
        api,
        "_get_failed_jobs",
        return_value=[{"job_id": "12345", "name": "x", "state": "FAILED"}],
    ):
        # No tracked job has failed; should be silent.
        api.check_job_health()


def test_verify_running_jobs_recognizes_registered(
    slurm_settings: SlurmManagerSettings,
):
    """A SLURM job whose name matches a server-known service name passes."""
    api = SlurmBatchApi(slurm_settings)
    job_name = f"{slurm_settings.job_name_prefix}-deadbeef"
    with patch.object(
        api,
        "_get_running_jobs",
        return_value=[{"job_id": "1", "name": job_name, "state": "RUNNING"}],
    ):
        api.verify_running_jobs({job_name})


def test_verify_running_jobs_raises_on_unregistered(
    slurm_settings: SlurmManagerSettings,
):
    """A SLURM job whose name is *not* in server_job_names triggers the error."""
    api = SlurmBatchApi(slurm_settings)
    job_name = f"{slurm_settings.job_name_prefix}-deadbeef"
    with patch.object(
        api,
        "_get_running_jobs",
        return_value=[{"job_id": "1", "name": job_name, "state": "RUNNING"}],
    ):
        with pytest.raises(JobNotFoundError, match=job_name):
            api.verify_running_jobs(set())


def test_verify_running_jobs_skips_non_managed_jobs(
    slurm_settings: SlurmManagerSettings,
):
    """Jobs without our prefix are ignored entirely."""
    api = SlurmBatchApi(slurm_settings)
    with patch.object(
        api,
        "_get_running_jobs",
        return_value=[{"job_id": "1", "name": "someone-elses-job", "state": "RUNNING"}],
    ):
        api.verify_running_jobs(set())


def test_jobs_pending_only_counts_tracked(slurm_settings: SlurmManagerSettings):
    api = SlurmBatchApi(slurm_settings)
    api.tracked_jobs.add("12345")
    with patch.object(
        api,
        "_get_pending_jobs",
        return_value=[{"job_id": "67890", "name": "x", "state": "PENDING"}],
    ):
        # The pending job is not in our tracked set
        assert api.jobs_pending() is False
    with patch.object(
        api,
        "_get_pending_jobs",
        return_value=[{"job_id": "12345", "name": "x", "state": "PENDING"}],
    ):
        assert api.jobs_pending() is True


def test_untrack_completed_jobs_removes_from_set(
    slurm_settings: SlurmManagerSettings,
):
    api = SlurmBatchApi(slurm_settings)
    api.tracked_jobs.update({"1", "2", "3"})
    with patch.object(
        api,
        "_get_completed_jobs",
        return_value=[
            {"job_id": "1", "name": "x", "state": "COMPLETED"},
            {"job_id": "2", "name": "y", "state": "COMPLETED"},
        ],
    ):
        api.untrack_completed_jobs()
    assert api.tracked_jobs == {"3"}


def test_untrack_completed_jobs_respects_setting(tmp_path: Path):
    template = tmp_path / "job-template.sh"
    template.write_text("#!/bin/bash\n#SBATCH --job-name={{ JOB_NAME }}\n")
    settings = SlurmManagerSettings(
        name="testmgr",
        logfile=None,
        max_compute_services=2,
        job_script_template=template,
        untrack_completed_jobs=False,
    )
    api = SlurmBatchApi(settings)
    api.tracked_jobs.add("1")
    with patch.object(
        api,
        "_get_completed_jobs",
        return_value=[{"job_id": "1", "name": "x", "state": "COMPLETED"}],
    ):
        api.untrack_completed_jobs()
    # Setting disabled -> nothing removed
    assert api.tracked_jobs == {"1"}


def test_submit_job_parses_sbatch_output(
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


def test_cancel_job_calls_scancel_and_untracks(
    slurm_settings: SlurmManagerSettings,
):
    api = SlurmBatchApi(slurm_settings)
    api.tracked_jobs.add("12345")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type(
            "R", (), {"stdout": "", "stderr": "", "returncode": 0}
        )
        api.cancel_job("12345")
    assert mock_run.call_args[0][0] == [
        slurm_settings.cancel_command,
        "12345",
    ]
    assert "12345" not in api.tracked_jobs


def test_cancel_command_is_in_settings(slurm_settings: SlurmManagerSettings):
    """Regression check: cancel_command must be exposed and used."""
    assert slurm_settings.cancel_command == "scancel"


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
        assert slurm_settings.job_name_prefix in content
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

    with patch.object(mgr.batch_api, "check_job_health"), patch.object(
        mgr.batch_api, "verify_running_jobs"
    ), patch.object(mgr.batch_api, "untrack_completed_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", side_effect=_fake_submit
    ):
        n = mgr.create_compute_services({"compute_service_ids": [], "num_tasks": 1})
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

    with patch.object(mgr.batch_api, "check_job_health"), patch.object(
        mgr.batch_api, "verify_running_jobs"
    ), patch.object(mgr.batch_api, "untrack_completed_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=False
    ), patch.object(
        mgr.batch_api, "submit_job", side_effect=_fake_submit
    ):
        mgr.create_compute_services({"compute_service_ids": [], "num_tasks": 1})
    # The script should still exist.
    assert submitted_paths[0].exists()
    submitted_paths[0].unlink()


def test_create_compute_services_skips_when_pending(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    mgr = _build_manager(slurm_settings, tmp_path)
    with patch.object(mgr.batch_api, "check_job_health"), patch.object(
        mgr.batch_api, "verify_running_jobs"
    ), patch.object(mgr.batch_api, "untrack_completed_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=True
    ), patch.object(
        mgr.batch_api, "submit_job"
    ) as submit:
        n = mgr.create_compute_services({"compute_service_ids": [], "num_tasks": 5})
    assert n == 0
    submit.assert_not_called()


def test_create_compute_services_strips_uuid_suffix_correctly(
    slurm_settings: SlurmManagerSettings, tmp_path: Path
):
    """server_job_names should preserve the full ``name`` (which may itself
    contain hyphens) by stripping only the trailing UUID hex suffix.
    """
    mgr = _build_manager(slurm_settings, tmp_path)

    # Service ID: "alchemiscale-12345678-abcd-1234-5678-90abcdef-deadbeef..."
    # The "name" portion contains 4 hyphens (UUID4 dashed form), and the
    # trailing hex is the random suffix. Only the final segment should be
    # stripped.
    service_name = f"{slurm_settings.job_name_prefix}-12345678-abcd-1234-5678-90abcdef"
    service_id = f"{service_name}-deadbeef0123456789abcdef01234567"

    captured = {}

    def _capture(server_job_names):
        captured["names"] = server_job_names

    with patch.object(mgr.batch_api, "check_job_health"), patch.object(
        mgr.batch_api, "verify_running_jobs", side_effect=_capture
    ), patch.object(mgr.batch_api, "untrack_completed_jobs"), patch.object(
        mgr.batch_api, "jobs_pending", return_value=True
    ):
        mgr.create_compute_services(
            {"compute_service_ids": [service_id], "num_tasks": 0}
        )

    assert captured["names"] == {service_name}
