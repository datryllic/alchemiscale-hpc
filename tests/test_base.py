"""Smoke tests for the abstract base interfaces and the backend registry."""

from pathlib import Path

import pytest

from alchemiscale_hpc import base


def test_abstract_classes_cannot_be_instantiated():
    """ABC enforcement prevents instantiating the base classes directly."""
    with pytest.raises(TypeError, match="abstract"):
        base.HPCBatchApi(settings=None)  # type: ignore[abstract]
    with pytest.raises(TypeError, match="abstract"):
        base.HPCManager(settings=None, service_settings_path=Path("/dev/null"))  # type: ignore[abstract]
    with pytest.raises(TypeError, match="abstract"):
        base.ScriptTemplateHPCManager(  # type: ignore[abstract]
            settings=None, service_settings_path=Path("/dev/null")
        )


def test_register_and_lookup_backend(monkeypatch):
    """Backends can be registered, listed, and retrieved."""
    # Snapshot and restore the registry so this test doesn't leak state.
    saved = dict(base._BACKENDS)
    monkeypatch.setattr(base, "_BACKENDS", dict(saved))

    class _Mgr:
        pass

    class _Settings:
        pass

    class _Api:
        pass

    base.register_backend("fake", _Mgr, _Settings, _Api)
    assert "fake" in base.list_backends()
    assert base.get_backend("fake") == (_Mgr, _Settings, _Api)


def test_get_backend_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        base.get_backend("definitely-not-a-real-backend")


def test_slurm_backend_self_registers():
    """Importing the SLURM subpackage registers it with the base registry."""
    # Importing through the lazy attribute should trigger registration.
    import alchemiscale_hpc

    _ = alchemiscale_hpc.SlurmManager
    assert "slurm" in base.list_backends()
    manager_cls, settings_cls, batch_api_cls = base.get_backend("slurm")
    assert manager_cls.__name__ == "SlurmManager"
    assert settings_cls.__name__ == "SlurmManagerSettings"
    assert batch_api_cls.__name__ == "SlurmBatchApi"


def test_settings_inheritance():
    """The settings hierarchy holds: Slurm < ScriptTemplate < HPCManagerSettings."""
    from alchemiscale_hpc.slurm import SlurmManagerSettings

    assert issubclass(SlurmManagerSettings, base.ScriptTemplateHPCManagerSettings)
    assert issubclass(base.ScriptTemplateHPCManagerSettings, base.HPCManagerSettings)
