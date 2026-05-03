"""PBS/Torque-specific compute manager for alchemiscale (coming soon).

To implement PBS support:

1. Create settings.py:
   - Define PBSManagerSettings(HPCManagerSettings)
   - Add PBS-specific configuration (queue, account, resource requirements)

2. Create manager.py:
   - Implement PBSBatchApi(HPCBatchApi)
     - Use qsub, qstat, qacct/tracejob commands
     - Parse PBS output to track job states
   - Implement PBSManager(HPCManager)
     - Generate PBS job scripts from template
     - Implement _create_job_script() method

3. Update cli.py:
   - Add PBS-specific commands under pbs_group

4. Create examples:
   - PBS job script template
   - Manager and service configuration files
   - README with setup instructions

See alchemiscale_hpc/slurm/ for a complete reference implementation.

Note: This should work for both PBS and Torque variants, with
      configuration options to handle differences.
"""

# Future imports:
# from .manager import PBSManager, PBSBatchApi
# from .settings import PBSManagerSettings
#
# __all__ = ["PBSManager", "PBSManagerSettings", "PBSBatchApi"]
