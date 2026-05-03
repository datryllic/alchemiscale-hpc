# LSF Examples (Coming Soon)

LSF support is not yet implemented. Contributions are welcome!

## Implementation Guide

To add LSF support, you'll need to:

1. **Implement the base interfaces** in `alchemiscale_hpc/lsf/`:
   - `LSFManagerSettings(HPCManagerSettings)` - Configuration
   - `LSFBatchApi(HPCBatchApi)` - Wrapper for LSF commands (bsub, bjobs, bacct)
   - `LSFManager(HPCManager)` - Main manager class

2. **Create example configurations**:
   - LSF job script template with `#BSUB` directives
   - Manager configuration YAML
   - Service configuration YAML

3. **Update the CLI** in `alchemiscale_hpc/cli.py`:
   - Implement the commands under the `lsf_group`

4. **Reference implementation**: See `alchemiscale_hpc/slurm/` for a complete working example

## LSF Commands Reference

The implementation should use these LSF commands:

- `bsub` - Submit batch jobs
- `bjobs` - Query job status
- `bkill` - Cancel jobs
- `bacct` - Query accounting information for completed jobs

## Contributing

Please submit pull requests to add LSF support! See the main README for contribution guidelines.
