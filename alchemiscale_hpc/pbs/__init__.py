"""PBS/Torque-specific compute manager for alchemiscale (coming soon).

To implement PBS support:

1. Create ``settings.py``:
   - Define ``PBSManagerSettings(ScriptTemplateHPCManagerSettings)``
   - Add PBS-specific configuration (queue, account, resource requirements)

2. Create ``manager.py``:
   - Implement ``PBSBatchApi(HPCBatchApi)``: wrap ``qsub``, ``qstat``,
     ``qdel``, ``qacct`` (or ``tracejob``); parse PBS output into the standard
     ``{job_id, name, state}`` dicts.
   - Implement ``PBSManager(ScriptTemplateHPCManager)``: provide
     ``_create_batch_api()`` and ``_create_job_script()``.

3. Register the backend in this ``__init__.py`` via
   ``alchemiscale_hpc.base.register_backend("pbs", PBSManager,
   PBSManagerSettings, PBSBatchApi)``. The CLI will pick it up automatically.

4. Create examples in ``examples/pbs/``:
   - PBS job script template (with ``#PBS`` directives)
   - Manager and service configuration files
   - README with setup instructions

See ``alchemiscale_hpc/slurm/`` for a complete reference implementation.

Note: This should support both PBS Pro and Torque variants, with config
      options to handle command differences.
"""

# Future implementation:
# from ..base import register_backend
# from .manager import PBSBatchApi, PBSManager
# from .settings import PBSManagerSettings
#
# register_backend("pbs", PBSManager, PBSManagerSettings, PBSBatchApi)
#
# __all__ = ["PBSManager", "PBSManagerSettings", "PBSBatchApi"]
