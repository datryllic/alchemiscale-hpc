# PBS/Torque Examples (Coming Soon)

PBS/Torque support is not yet implemented. Contributions are welcome!

## Implementation Guide

To add PBS support, you'll need to:

1. **Implement the base interfaces** in `alchemiscale_hpc/pbs/`:
   - `PBSManagerSettings(HPCManagerSettings)` - Configuration
   - `PBSBatchApi(HPCBatchApi)` - Wrapper for PBS commands (qsub, qstat, qacct)
   - `PBSManager(HPCManager)` - Main manager class

2. **Create example configurations**:
   - PBS job script template with `#PBS` directives
   - Manager configuration YAML
   - Service configuration YAML

3. **Update the CLI** in `alchemiscale_hpc/cli.py`:
   - Implement the commands under the `pbs_group`

4. **Reference implementation**: See `alchemiscale_hpc/slurm/` for a complete working example

## PBS Commands Reference

The implementation should use these PBS commands:

- `qsub` - Submit batch jobs
- `qstat` - Query job status
- `qdel` - Cancel jobs
- `qacct` or `tracejob` - Query accounting information for completed jobs

## PBS vs Torque

This implementation should support both PBS and Torque variants. Consider:

- Configuration options to handle command differences
- Different output formats between PBS Pro and Torque
- Testing on both systems if possible

## Contributing

Please submit pull requests to add PBS support! See the main README for contribution guidelines.
