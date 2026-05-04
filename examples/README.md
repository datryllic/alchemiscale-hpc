# alchemiscale-hpc Examples

This directory contains example configurations for different HPC queueing systems.

## Supported Systems

- **[SLURM](slurm/)**: Fully implemented with autoscaling manager
- **LSF**: Coming soon (contributions welcome!)
- **PBS/Torque**: Coming soon (contributions welcome!)

## Directory Structure

```
examples/
├── slurm/              # SLURM-specific examples
│   ├── README.md
│   ├── job-template.sh
│   ├── manager-config.yml
│   └── service-config.yml
├── lsf/                # LSF examples (future)
│   └── README.md
└── pbs/                # PBS examples (future)
    └── README.md
```

## Quick Start

1. Choose your queueing system directory (e.g., `slurm/`)
2. Follow the README in that directory for system-specific setup
3. Customize the configuration files for your HPC environment
4. Start the manager using the appropriate CLI command:
   - SLURM: `alchemiscale-hpc slurm start -c config.yml -s service.yml`
   - LSF: `alchemiscale-hpc lsf start -c config.yml -s service.yml` (coming soon)
   - PBS: `alchemiscale-hpc pbs start -c config.yml -s service.yml` (coming soon)

## Contributing

To add support for a new queueing system:

1. Create a new subdirectory: `examples/<system>/`
2. Implement the `ScriptTemplateHPCManager` and `HPCBatchApi` interfaces in
   `alchemiscale_hpc/<system>/` (see `alchemiscale_hpc/base.py`).
3. Self-register the backend by calling
   `register_backend(name, Manager, Settings, BatchApi)` in your subpackage's
   `__init__.py`. The CLI exposes the new subcommand group automatically — no
   changes to `cli.py` required.
4. Add example configurations under `examples/<system>/`.
5. Submit a pull request!

See `alchemiscale_hpc/slurm/` for a complete reference implementation.
