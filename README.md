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

Support for these systems is coming soon. The CLI groups (`alchemiscale-hpc lsf ...`,
`alchemiscale-hpc pbs ...`) accept the same options as SLURM and report a
"not yet implemented" message until a backend is registered. See
`examples/lsf/README.md` and `examples/pbs/README.md` for implementation guides.

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

The package uses a modular architecture with two abstraction layers:

1. **Bare base interfaces** (`base.py`): Enforced via `abc.ABC`.
   - `HPCManager`: minimal manager contract (`_create_batch_api`,
     `create_compute_services`). Extends alchemiscale's `ComputeManager`.
   - `HPCBatchApi`: abstract interface for batch system commands
     (`submit_job`, `get_jobs`, `check_job_health`,
     `clear_successful_jobs`, `clear_failed_jobs`, ...).
   - `HPCManagerSettings`: settings every backend needs (`name`,
     `job_name_prefix`, `max_submit_per_cycle`, ...).

2. **Script-template convenience layer** (`base.py`): the workflow shared
   by SLURM/LSF/PBS.
   - `ScriptTemplateHPCManager(HPCManager)`: implements
     `create_compute_services` as "render template → submit → cleanup".
     Subclasses only implement `_create_batch_api` and `_create_job_script`.
   - `ScriptTemplateHPCManagerSettings(HPCManagerSettings)`: adds
     `job_script_template`, `job_script_dir`, `keep_job_scripts`,
     `cleanup_successful_jobs`, `cleanup_failed_jobs`.

3. **Backend registry**: each backend self-registers via
   `register_backend(name, Manager, Settings, BatchApi)` in its package
   `__init__.py`. The CLI builds its subcommand groups from the registry
   automatically — adding a new backend requires no CLI edits.

4. **System-specific implementations**: each queueing system has its own
   subpackage (`alchemiscale_hpc/<system>/`), wraps system commands
   (sbatch/bsub/qsub, squeue/bjobs/qstat, ...), and provides a job-script
   template for users to copy and customize.

5. **Common autoscaling logic** is inherited from alchemiscale's
   `ComputeManager`: request instructions from the server, scale based on
   task availability, report status and saturation.

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
   - `<System>ManagerSettings(ScriptTemplateHPCManagerSettings)` in `settings.py`
   - `<System>BatchApi(HPCBatchApi)` in `manager.py`
   - `<System>Manager(ScriptTemplateHPCManager)` in `manager.py`
3. **Register the backend** in `alchemiscale_hpc/<system>/__init__.py`:
   ```python
   from ..base import register_backend
   register_backend("<system>", <System>Manager, <System>ManagerSettings, <System>BatchApi)
   ```
   The CLI subcommand group is generated automatically.
4. **Create examples**: Add configuration files and documentation in `examples/<system>/`
5. **(Optional) Lazy export**: Add to the `_LAZY` map in
   `alchemiscale_hpc/__init__.py` if you want top-level access via
   `alchemiscale_hpc.<System>Manager`.
6. **Test thoroughly**: Ensure autoscaling works on your target HPC system.

See `alchemiscale_hpc/slurm/` for a complete reference implementation.

## Tests

A small smoke-test suite covers the abstract interfaces, the SLURM
implementation, and the CLI registry:

```bash
pip install -e .[test]
pytest -v
```

Tests use mocked subprocess calls and do not require an actual SLURM cluster.

## License

MIT
