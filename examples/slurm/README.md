# SLURM Examples for alchemiscale-hpc

This directory contains example configuration files for using the SLURM compute manager.

## Files

- **slurm-job-template.sh**: Template SLURM batch script that the manager uses to submit compute service jobs
- **manager-config.yml**: Configuration for the SlurmManager (autoscaling settings)
- **service-config.yml**: Configuration for compute services that will be launched

## Quick Start

### 1. Setup

First, install alchemiscale-hpc:

```bash
pip install -e /path/to/alchemiscale-hpc
```

### 2. Customize Configuration Files

Copy the example files and customize them for your HPC environment:

```bash
cp examples/slurm-job-template.sh my-job-template.sh
cp examples/manager-config.yml my-manager-config.yml
cp examples/service-config.yml my-service-config.yml
```

**Edit `my-job-template.sh`:**
- Adjust `#SBATCH` directives for your HPC system (CPUs, GPUs, memory, time limits)
- Update module load commands
- Set correct conda environment path
- Update `SERVICE_CONFIG` path to point to your service config file

**Edit `my-manager-config.yml`:**
- Set `name` to a unique identifier for this manager
- Set `job_script_template` to the path of your job template
- Configure `partition`, `account`, `qos` for your SLURM system
- Set `max_compute_services` based on your resource allocation
- Adjust `sleep_interval` and `max_submit_per_cycle` for your scaling needs
- Set `logfile` path

**Edit `my-service-config.yml`:**
- Set `api_url` to your alchemiscale compute API endpoint
- Set `identifier` and `key` with your compute identity credentials
- Set `name` to identify your HPC cluster
- Configure `shared_basedir` and `scratch_basedir` paths
- Set `scopes` to limit which tasks to claim
- Adjust `logfile` path

### 3. Create Required Directories

Make sure the log and scratch directories exist:

```bash
mkdir -p /path/to/logs
mkdir -p /scratch/alchemiscale/{shared,scratch,cache,logs}
```

### 4. Start the Manager

Start the manager process (typically in a persistent session like tmux or screen):

```bash
# In a tmux/screen session:
alchemiscale-hpc slurm start \
    -c my-manager-config.yml \
    -s my-service-config.yml
```

The manager will:
1. Register with the alchemiscale compute API
2. Periodically check for available tasks
3. Submit SLURM jobs when tasks are waiting and capacity is available
4. Monitor job health and cleanup completed/failed jobs
5. Report status back to the server

### 5. Monitor the Manager

Check manager logs:

```bash
tail -f /path/to/logs/manager.log
```

View SLURM jobs:

```bash
alchemiscale-hpc slurm show-jobs -c my-manager-config.yml
```

Or use standard SLURM commands:

```bash
squeue -u $USER
```

### 6. Stop the Manager

Press `Ctrl+C` in the manager process to gracefully shutdown. The manager will:
- Stop submitting new jobs
- Deregister from the server
- Exit cleanly

## Advanced Usage

### Clear Error Status

If the manager gets stuck in ERROR state:

```bash
alchemiscale-hpc slurm clear-error \
    -c my-manager-config.yml \
    -s my-service-config.yml
```

### Cleanup Jobs

Clean up failed jobs from tracking:

```bash
alchemiscale-hpc slurm cleanup -c my-manager-config.yml --failed
```

Clean up completed jobs:

```bash
alchemiscale-hpc slurm cleanup -c my-manager-config.yml --completed
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ alchemiscale Server                                         │
│ ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│ │ Neo4j StateStore│  │ S3 ObjectStore│ │ Compute API    │  │
│ └─────────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             ▲
                             │ HTTP/JWT
                             │
┌────────────────────────────┼─────────────────────────────────┐
│ HPC Cluster                │                                 │
│                            │                                 │
│  ┌─────────────────────────┴──────────┐                     │
│  │ SlurmManager (long-running)        │                     │
│  │ - Checks for tasks                 │                     │
│  │ - Submits SLURM jobs               │                     │
│  │ - Monitors job health              │                     │
│  └────────────────────────────────────┘                     │
│                    │                                         │
│                    │ sbatch                                  │
│                    ▼                                         │
│  ┌──────────────────────────────────────────┐               │
│  │ SLURM Queue                              │               │
│  │  ┌──────────────┐  ┌──────────────┐     │               │
│  │  │ Compute      │  │ Compute      │ ... │               │
│  │  │ Service (job)│  │ Service (job)│     │               │
│  │  │ - Claim tasks│  │ - Claim tasks│     │               │
│  │  │ - Execute    │  │ - Execute    │     │               │
│  │  │ - Push results│ │ - Push results│    │               │
│  │  └──────────────┘  └──────────────┘     │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Scaling Logic

1. **Manager requests instruction** from compute API with scopes
2. **Server responds** with:
   - List of currently active compute service IDs
   - Number of waiting tasks in those scopes
3. **Manager decides** based on:
   - Current services < max_compute_services?
   - Tasks waiting > 0?
   - No jobs currently pending (starting up)?
4. **If scaling up**: Submit SLURM job(s)
5. **SLURM job**:
   - Starts compute service
   - Service registers with API
   - Claims and executes tasks
   - Deregisters when done (max tasks/time reached or no more tasks)
6. **Manager tracks** job IDs and cleans up completed/failed jobs

### Job Lifecycle

```
[Manager submits] → [PENDING] → [RUNNING] → [COMPLETED]
                                     │
                                     └──→ [FAILED]
```

- **PENDING**: Job is queued in SLURM, waiting for resources
- **RUNNING**: Job is executing, compute service is active
- **COMPLETED**: Job finished successfully, service deregistered
- **FAILED**: Job failed, manager raises alert

## Troubleshooting

### Manager not submitting jobs

Check:
- Are tasks actually waiting? (query the alchemiscale server)
- Is `max_compute_services` reached?
- Are there jobs in PENDING state? (manager waits for them to start)
- Check manager logs for errors

### Jobs failing immediately

Check:
- SLURM job logs: `logs/alchemiscale-<jobid>.err`
- Verify module loads and conda activation work
- Check paths to service config file
- Verify compute identity credentials

### Jobs not claiming tasks

Check:
- Service config `scopes` match available tasks
- Compute identity has permissions for those scopes
- API URL is correct and accessible from compute nodes
- Network connectivity from compute nodes

### Jobs running but no results

Check:
- Service logs for task execution errors
- Scratch space is available and writable
- GPU is accessible if required
- OpenMM/molecular dynamics environment is properly configured
