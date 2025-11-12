# alchemiscale-hpc

Alchemiscale tools for use with HPC systems.

This package provides a SLURM-based compute manager for autoscaling alchemiscale compute services on HPC clusters.

## Features

- **Autoscaling**: Automatically submit SLURM jobs based on task availability
- **Resource Management**: Configure resource limits and manage concurrent jobs
- **Health Monitoring**: Track job status and handle failures gracefully
- **Flexible Configuration**: YAML-based configuration for both manager and services
- **CLI Tools**: Command-line interface for management and monitoring

## Installation

```bash
pip install -e .
```

## Quick Start

1. Copy example configuration files:

```bash
cp examples/slurm-job-template.sh my-job-template.sh
cp examples/manager-config.yml my-manager-config.yml
cp examples/service-config.yml my-service-config.yml
```

2. Customize the configuration files for your HPC environment (see `examples/README.md`)

3. Start the manager:

```bash
alchemiscale-hpc manager start -c my-manager-config.yml -s my-service-config.yml
```

## Documentation

See `examples/README.md` for detailed setup instructions and usage examples.

## How It Works

The SlurmManager monitors task availability from the alchemiscale server and automatically submits SLURM batch jobs to execute compute services when tasks are waiting and capacity is available.

Each SLURM job runs an alchemiscale compute service that:
- Registers with the alchemiscale compute API
- Claims and executes tasks
- Pushes results back to the server
- Deregisters when complete or max time/tasks reached

The manager handles job lifecycle, health monitoring, and cleanup of completed/failed jobs.

## CLI Commands

```bash
# Start the manager
alchemiscale-hpc manager start -c manager-config.yml -s service-config.yml

# Clear error status
alchemiscale-hpc manager clear-error -c manager-config.yml -s service-config.yml

# Show SLURM jobs
alchemiscale-hpc slurm show-jobs -c manager-config.yml

# Cleanup completed/failed jobs
alchemiscale-hpc slurm cleanup -c manager-config.yml --failed --completed
```

## Architecture

The SlurmManager follows the same pattern as K8SManager in alchemiscale-k8s:

- Extends `ComputeManager` base class from alchemiscale
- Implements platform-specific job submission via `SlurmBatchApi`
- Uses SLURM commands (sbatch, squeue, sacct) to manage job lifecycle
- Integrates with alchemiscale compute API for task coordination

## License

MIT
