"""LSF-specific compute manager for alchemiscale (coming soon).

To implement LSF support:

1. Create ``settings.py``:
   - Define ``LSFManagerSettings(ScriptTemplateHPCManagerSettings)``
   - Add LSF-specific configuration (queue, project, resource requirements)

2. Create ``manager.py``:
   - Implement ``LSFBatchApi(HPCBatchApi)``: wrap ``bsub``, ``bjobs``,
     ``bkill``, ``bacct``; parse LSF output into the standard
     ``{job_id, name, state}`` dicts.
   - Implement ``LSFManager(ScriptTemplateHPCManager)``: provide
     ``_create_batch_api()`` and ``_create_job_script()``.

3. Register the backend in this ``__init__.py`` via
   ``alchemiscale_hpc.base.register_backend("lsf", LSFManager,
   LSFManagerSettings, LSFBatchApi)``. The CLI will pick it up automatically.

4. Create examples in ``examples/lsf/``:
   - LSF job script template (with ``#BSUB`` directives)
   - Manager and service configuration files
   - README with setup instructions

See ``alchemiscale_hpc/slurm/`` for a complete reference implementation.
"""

# Future implementation:
# from ..base import register_backend
# from .manager import LSFBatchApi, LSFManager
# from .settings import LSFManagerSettings
#
# register_backend("lsf", LSFManager, LSFManagerSettings, LSFBatchApi)
#
# __all__ = ["LSFManager", "LSFManagerSettings", "LSFBatchApi"]
