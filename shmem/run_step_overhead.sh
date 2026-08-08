#!/bin/bash
# Sweep the shared-memory step-overhead microbenchmark to the full 96 threads on an EXCLUSIVE Genoa
# node.
#
#   sbatch run_step_overhead.sh
#
# Why not just run it on the login node: the first sweep was capped at 32 threads and shared with
# other users, and the two most interesting rows were exactly at 32 -- the FORK overhead floor jumping
# to 0.29 ms (17x POOL's 0.017 ms) and the 620-active case reversing. Both are where contention would
# also show up, so they need an exclusive node to be trusted, and both need 96 threads to be relevant
# to the real machine.
#
# What the sweep decides:
#   1. the overhead FLOOR at 96 threads -- the number the whole rewrite rests on. MPI's per-step floor
#      is ~8.85 ms; if a thread pool's is microseconds, the thesis holds.
#   2. FORK vs POOL -- whether a parallel region per step (GIZMO's current pattern) is the thing that
#      cost Stage 0 its 7.4x.
#   3. ADAPT -- whether capping width at nactive/40 beats using every thread. At 620 active elements
#      96 threads is 6.5 items each, and the 32-thread run already reversed.
#SBATCH --job-name=shmem_overhead
#SBATCH --partition=cca
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=overhead-%j.out
#SBATCH --error=overhead-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module --force purge
module load modules gcc

# Rebuild on the compute node: -march=native must target Genoa, not the login node's architecture.
g++ -O2 -fopenmp -march=native -o step_overhead step_overhead.cc
echo "=== built on $(hostname), $(nproc) cores ==="
g++ --version | head -1

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "=== placement: $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}') ==="
echo ""
./step_overhead 96
