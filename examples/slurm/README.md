# SLURM Examples for alchemiscale-hpc

This directory contains example configuration files for using the SLURM compute manager.

## Files

- **job-template.sh**: Template SLURM batch script that the manager uses to submit compute service jobs
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
cp examples/slurm/job-template.sh my-job-template.sh
cp examples/slurm/manager-config.yml my-manager-config.yml
cp examples/slurm/service-config.yml my-service-config.yml
```

**Edit `my-job-template.sh`:**
- Adjust `#SBATCH` directives for your HPC system (CPUs, GPUs, memory, time limits)
- Update module load commands
- Set correct conda environment path
- Update `SERVICE_CONFIG` path to point to your service config file

**Edit `my-manager-config.yml`:**
- Set `name` for this manager. SLURM job names are built as
  `<name>.<uuid_hex>`, so `name` must start with a letter and contain only
  letters, digits, `_`, `.`, and `:` (no `-`). The manager recognizes its jobs
  by name, so managers running under the same cluster user need different
  names, even if they use different alchemiscale servers. Managers under
  different users can share a name
- Set `job_script_template` to the path of your job template
- Configure `partition`, `account`, `qos` for your SLURM system; the manager passes them to `sbatch` on the command line, overriding matching `#SBATCH` directives in the template
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
4. Log failed jobs with their reasons, and stop tracking completed and
   failed jobs
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

Add `--mine` to show only the jobs submitted by the manager named in the
config (job names `<name>.<uuid_hex>`).

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

### Failed Jobs

A failed job (`FAILED`, `TIMEOUT`, `CANCELLED`, `NODE_FAIL`, `PREEMPTED`,
`OUT_OF_MEMORY`, `BOOT_FAIL`, `DEADLINE`) does not stop the manager. Each
cycle that checks its jobs, the manager logs every newly failed job at ERROR
with its `sacct` state and exit code (and last pending reason, if any), for
example:

```
Batch job 4242 failed: CANCELLED by 1000, exit code 0, signal 15
```

and stops tracking it. The manager stops with `JobFailureError` only if
failed jobs are found in more than `max_consecutive_failed_cycles` (default
5) checks with no sign of a healthy job in between: one of its jobs
completing, or one of its compute services registered with the server.
Checks that only see pending jobs neither count nor reset. So routine
preemptions and timeouts in a working fleet never stop the manager, while
jobs that fail without ever registering do, even if they pend first. After
a change that breaks the template, an older service of this manager that is
still registered keeps resetting the count until it ends (up to its
walltime); only then does the limit apply. The new jobs' failures are
logged at ERROR throughout.

Jobs you cancel yourself with `scancel` are recorded as `CANCELLED` and count
as failed too. Cancelling several while none of the manager's compute services
is registered can stop it; stop the manager first if you need to cancel many
of its jobs.

Without SLURM accounting (`sacct`), the manager cannot see failed jobs at all:
it logs a WARNING at startup, and `max_consecutive_failed_cycles` never stops
it.

Your job script must exit with the compute service's exit status, as the
example template does: a job that exits 0 is recorded as `COMPLETED`, which
counts as a sign of health, so a service that never started would otherwise
keep the manager resubmitting.

Likewise, it stops with `JobSubmissionError` if every `sbatch` fails in more
than that many cycles while none of its compute services is registered;
failures at a submit limit (e.g. `MaxSubmitJobsPerUser`) with a running
fleet do not count. The error includes `sbatch`'s message. Either error
usually means a broken job script template or settings (partition, account,
QoS). With a short `sleep_interval`, raise the limit so that a brief
`slurmctld` outage does not stop the manager.

A running job that is not registered with the server, past
`job_registration_grace_period` from when it started, is logged at WARNING
and checked again next cycle: a job can stay RUNNING briefly after its
compute service deregisters. The manager stops with `JobNotFoundError` only
if the job is still unregistered on a later check at least
`job_registration_grace_period` seconds after it was first found.

### Show the Manager's Jobs

List the manager's queued and running jobs:

```bash
alchemiscale-hpc slurm show-jobs -c my-manager-config.yml --mine
```

Add `--finished` to also list the jobs that finished in the last 7 days, with
their final state, from `sacct`:

```bash
alchemiscale-hpc slurm show-jobs -c my-manager-config.yml --mine --finished
```

The manager's jobs are found by name, with the same queries the manager uses
when it starts. SLURM keeps finished jobs only as accounting records, so there
is nothing to clean up. To keep job scripts after submission (useful for
debugging), set `keep_job_scripts: true` in the manager config.

### Restarting After an Error

On startup, the manager recovers its jobs that are still in the SLURM queue;
if `squeue` fails, it does not start. It then recovers its jobs that
finished in the last 7 days from `sacct`, and logs one WARNING with the
number that failed (including ones already reported before the restart),
with each job and its reason at INFO. These do not count toward
`max_consecutive_failed_cycles`. If `sacct` fails (e.g. accounting is
disabled), it logs a WARNING and starts anyway.

Fix the cause named in the manager log before restarting; if the manager is
still in ERROR state on the server, clear it first (see
[Clear Error Status](#clear-error-status)).

A restart does not forget a running job that is not registered with the
server: it is recovered and, after a check and a re-check, stops the manager
with `JobNotFoundError` again, until the job ends or you cancel it with
`scancel`.

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
6. **Manager tracks** job IDs, logs failed jobs with their reasons, and
   stops tracking completed and failed jobs. On startup, it recovers its jobs
   from `squeue` and `sacct` by job name, so a restarted manager does not
   submit on top of jobs that are still pending. This is why managers under the
   same cluster user need different names.

### Job Lifecycle

```
[Manager submits] → [PENDING] → [RUNNING] → [COMPLETED]
                                     │
                                     └──→ [FAILED]
```

- **PENDING**: Job is queued in SLURM, waiting for resources
- **RUNNING**: Job is executing, compute service is active
- **COMPLETED**: Job finished successfully, service deregistered
- **FAILED** (or `TIMEOUT`, `CANCELLED`, `NODE_FAIL`, `PREEMPTED`,
  `OUT_OF_MEMORY`, `BOOT_FAIL`, `DEADLINE`): manager
  logs the reason and carries on; it stops only if jobs fail cycle after cycle
  (see [Failed Jobs](#failed-jobs))

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
