"""LSF-specific compute manager for alchemiscale (coming soon).

To implement LSF support:

1. Create settings.py:
   - Define LSFManagerSettings(HPCManagerSettings)
   - Add LSF-specific configuration (queue, project, resource requirements)

2. Create manager.py:
   - Implement LSFBatchApi(HPCBatchApi)
     - Use bsub, bjobs, bacct commands
     - Parse LSF output to track job states
   - Implement LSFManager(HPCManager)
     - Generate LSF job scripts from template
     - Implement _create_job_script() method

3. Update cli.py:
   - Add LSF-specific commands under lsf_group

4. Create examples:
   - LSF job script template
   - Manager and service configuration files
   - README with setup instructions

See alchemiscale_hpc/slurm/ for a complete reference implementation.
"""

# Future imports:
# from .manager import LSFManager, LSFBatchApi
# from .settings import LSFManagerSettings
#
# __all__ = ["LSFManager", "LSFManagerSettings", "LSFBatchApi"]
