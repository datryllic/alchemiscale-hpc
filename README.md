# alchemiscale-hpc

Alchemiscale tools for use with HPC systems.

This package provides compute managers for autoscaling alchemiscale compute services on various HPC queueing systems.

## Supported Queueing Systems

- **✅ SLURM** - Fully implemented
- **⏳ LSF** - Coming soon (contributions welcome!)
- **⏳ PBS/Torque** - Coming soon (contributions welcome!)

## Features

- **Multi-System Support**: Modular architecture for multiple queueing systems
- **Autoscaling**: Automatically submit jobs based on task availability
- **Resource Management**: Configure resource limits and manage concurrent jobs
- **Health Monitoring**: Track job status and handle failures gracefully
- **Flexible Configuration**: YAML-based configuration for both manager and services
- **CLI Tools**: System-specific command-line interfaces

## Installation

```bash
pip install -e .
```

## Quick Start

### SLURM

1. Copy example configuration files:

```bash
cp examples/slurm/job-template.sh my-job-template.sh
cp examples/slurm/manager-config.yml my-manager-config.yml
cp examples/slurm/service-config.yml my-service-config.yml
```

2. Customize the configuration files for your HPC environment (see `examples/slurm/README.md`)

3. Start the manager:

```bash
alchemiscale-hpc slurm start -c my-manager-config.yml -s my-service-config.yml
```

### LSF / PBS

Support for these systems is coming soon. See `examples/lsf/README.md` and `examples/pbs/README.md` for implementation guides.

## Documentation

- **SLURM**: See `examples/slurm/README.md` for detailed setup instructions
- **LSF**: See `examples/lsf/README.md` for implementation guide
- **PBS**: See `examples/pbs/README.md` for implementation guide
- **Base API**: See `alchemiscale_hpc/base.py` for interface documentation

## CLI Commands

### SLURM

```bash
# Start the manager
alchemiscale-hpc slurm start -c manager-config.yml -s service-config.yml

# Clear error status
alchemiscale-hpc slurm clear-error -c manager-config.yml -s service-config.yml

# Show SLURM jobs
alchemiscale-hpc slurm show-jobs -c manager-config.yml

# Cleanup completed/failed jobs
alchemiscale-hpc slurm cleanup -c manager-config.yml --failed --completed
```

### LSF (Coming Soon)

```bash
# Start the manager
alchemiscale-hpc lsf start -c manager-config.yml -s service-config.yml
```

### PBS (Coming Soon)

```bash
# Start the manager
alchemiscale-hpc pbs start -c manager-config.yml -s service-config.yml
```

## Architecture

```
alchemiscale-hpc/
├── alchemiscale_hpc/
│   ├── base.py              # Base classes and interfaces
│   ├── slurm/               # SLURM implementation
│   │   ├── manager.py       # SlurmManager, SlurmBatchApi
│   │   └── settings.py      # SlurmManagerSettings
│   ├── lsf/                 # LSF implementation (future)
│   ├── pbs/                 # PBS implementation (future)
│   └── cli.py               # Command-line interface
└── examples/
    ├── slurm/               # SLURM examples
    ├── lsf/                 # LSF examples (future)
    └── pbs/                 # PBS examples (future)
```

### Design Principles

The package uses a modular architecture:

1. **Base Classes** (`base.py`): Define interfaces that all queueing systems must implement
   - `HPCManager`: Base manager class extending alchemiscale's `ComputeManager`
   - `HPCBatchApi`: Abstract interface for batch system commands
   - `HPCManagerSettings`: Common configuration options

2. **System-Specific Implementations**: Each queueing system (SLURM, LSF, PBS) has its own module:
   - Extends base classes with system-specific logic
   - Wraps system commands (sbatch/bsub/qsub, squeue/bjobs/qstat, etc.)
   - Implements job script generation

3. **Common Autoscaling Logic**: Inherited from alchemiscale's `ComputeManager`:
   - Request instructions from alchemiscale server
   - Scale up/down based on task availability
   - Report status and saturation metrics

## How It Works

The manager monitors task availability from the alchemiscale server and automatically submits batch jobs to execute compute services when tasks are waiting and capacity is available.

Each batch job runs an alchemiscale compute service that:
- Registers with the alchemiscale compute API
- Claims and executes tasks
- Pushes results back to the server
- Deregisters when complete or max time/tasks reached

The manager handles job lifecycle, health monitoring, and cleanup of completed/failed jobs.

## Contributing

Contributions are welcome, especially for adding support for new queueing systems!

### Adding a New Queueing System

1. **Create system directory**: `alchemiscale_hpc/<system>/`
2. **Implement interfaces**:
   - `<System>ManagerSettings(HPCManagerSettings)` in `settings.py`
   - `<System>BatchApi(HPCBatchApi)` in `manager.py`
   - `<System>Manager(HPCManager)` in `manager.py`
3. **Add CLI commands**: Update `cli.py` with system-specific subcommands
4. **Create examples**: Add configuration files and documentation in `examples/<system>/`
5. **Update imports**: Add to `alchemiscale_hpc/__init__.py`
6. **Test thoroughly**: Ensure autoscaling works on your target HPC system

See `alchemiscale_hpc/slurm/` for a complete reference implementation.

## License

MIT
