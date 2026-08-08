#!/bin/bash
# Time test/sedov on an EXCLUSIVE Genoa node, which is the only way to get a performance number
# worth quoting: the login/workstation runs so far were capped at 32 shared cores, and the whole
# point of this engine is what one shared-memory process does with a WHOLE node.
#
#   sbatch run_sedov.sh
#
# Sedov is the right benchmark for individual timesteps. 2,097,152 cells, and after the blast forms
# only a few hundred are on the deepest bin, so a global timestep costs ~3.3 s/sync against ~0.03 s
# for the individual-timestep scheme. It also has an exact solution, so the run is scored, not just
# timed -- a fast wrong answer is easy to produce here.
#
# Sweeps thread count so the number can be read as a scaling curve rather than a single point.
#SBATCH --job-name=shmem_sedov
#SBATCH --partition=cca
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=sedov-%j.out
#SBATCH --error=sedov-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module --force purge
module load modules gcc openmpi hdf5

# Rebuild on the compute node: -march=native must target Genoa, not the submit host.
make -B GIZMO
echo "=== built on $(hostname), $(nproc) cores ==="
g++ --version | head -1

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# A Genoa node is multi-socket, and every bulk array here is first-touched by ONE thread while the
# ICs are read -- so without interleaving, all of it lands on a single NUMA domain and every thread
# on the other sockets pays remote-memory latency for the whole run. That penalty grows with thread
# count, which is precisely the regime this engine exists to win in.
NUMA_PREFIX=""
if command -v numactl >/dev/null 2>&1; then
    NUMA_PREFIX="numactl --interleave=all"
    echo "=== NUMA topology ==="; numactl --hardware | head -8
fi

cd ../test/sedov
# ICs are fetched by the pytest harness; if this is a fresh checkout, run the test once on a login
# node first, or uncomment:
# wget -q http://www.tapir.caltech.edu/~phopkins/sims/sedov_ics.hdf5

for THREADS in 96 48 24 12; do
    rm -rf output
    export OMP_NUM_THREADS=$THREADS
    echo ""
    echo "############ OMP_NUM_THREADS=$THREADS ############"
    # Write the raw log per thread count AND tee it through, unbuffered. Piping straight into
    # `grep | tail` buffers the whole pass, so the job looks hung for ten minutes and there is no
    # way to tell a slow run from a stuck one -- which is exactly when you want the output.
    stdbuf -oL -eL /usr/bin/time -f "wall %e s   user %U s   maxrss %M kB" \
        $NUMA_PREFIX mpirun -np 1 ../../shmem/GIZMO sedov.params 0 2>&1 | \
        stdbuf -oL tee "sedov_omp${THREADS}.log" | \
        stdbuf -oL grep -E "^shmem-GIZMO|^done:|^  step |wall "
done

echo ""
echo "=== scoring the 96-thread result against the exact solution ==="
export OMP_NUM_THREADS=96
rm -rf output
cd ../..
export GIZMO_PREBUILT=$PWD/shmem/GIZMO PYTHONPATH=$PWD/python_src
python -m pytest test/sedov -q
