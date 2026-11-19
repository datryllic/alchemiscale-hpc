#!/bin/bash
#SBATCH --job-name={{ JOB_NAME }}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/alchemiscale-%j.out
#SBATCH --error=logs/alchemiscale-%j.err

# This is a template SLURM job script for alchemiscale compute services.
# The manager will replace the placeholders listed below with actual values.
# It warns at startup about any other double-brace token, which would be left
# in the script unchanged.
#
# Required placeholders (the manager refuses to start without them):
#   {{ JOB_NAME }}            - Unique job name (auto-generated)
#   {{ COMPUTE_MANAGER_ID }}  - Manager ID for provenance tracking
#
# The manager passes the `partition`, `account` and `qos` settings to sbatch on
# the command line, where they override any matching directives here.
#
# Customize the resource requests above (#SBATCH directives) for your HPC system.

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Compute Manager ID: {{ COMPUTE_MANAGER_ID }}"

# Load required modules (customize for your HPC system)
# Example:
# module load cuda/12.0
# module load python/3.11
# module load anaconda3

# Activate conda environment
# Customize this for your environment setup
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate alchemiscale-compute

# Set environment variables
export OPENMM_CPU_THREADS=$SLURM_CPUS_PER_TASK

# Path to compute service settings YAML file
# This should be created separately and contain all the necessary configuration
# for the compute service (API URL, credentials, scopes, etc.)
SERVICE_CONFIG="/path/to/compute-service-settings.yaml"

# Start the alchemiscale compute service.
#
# `--name "{{ JOB_NAME }}"` overrides the `name` field in the service config so
# that the registered ComputeServiceID matches the SLURM job name. This is
# REQUIRED for the manager's check that running jobs are registered.
#
# `--compute-manager-id "{{ COMPUTE_MANAGER_ID }}"` records the manager that
# launched this service for provenance.
#
# The service will:
#   1. Register with the compute API
#   2. Claim and execute tasks
#   3. Push results back to the API
#   4. Deregister when done or max time/tasks reached
alchemiscale compute synchronous \
    -c "$SERVICE_CONFIG" \
    --name "{{ JOB_NAME }}" \
    --compute-manager-id "{{ COMPUTE_MANAGER_ID }}"
# Capture the service's exit status right away. The job must exit with it:
# SLURM records a job that exits 0 as COMPLETED, which the manager takes as
# a sign of health, so a service that failed to start (command not found,
# wrong environment, bad service config) must not end the job with status 0.
status=$?

# Job completion
echo "End Time: $(date)"
echo "Compute service exited with status $status"
exit $status
