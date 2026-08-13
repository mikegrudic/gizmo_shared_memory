# GIZMO-SHMEM

A pure-OpenMP shared-memory rewrite of GIZMO's MFM hydro + tree gravity + STARFORGE sink
machinery, for single-node runs. One process, no domain decomposition, no MPI communication --
MPI is initialised only so `mpicxx`-linked HDF5 and the harness stay happy. The design goal is
fidelity to the reference implementation (the MPI GIZMO in this repo's parent directories):
when this engine and the reference disagree, the reference is right until proven otherwise.

Current scope: MFM hydro (`HYDRO_MESHLESS_FINITE_MASS`), Barnes-Hut gravity with adaptive
softening and the tidal-tensor timestep criterion, individual (power-of-two) timesteps with the
Saitoh-Makino limiter, STARFORGE sink formation / gravitational-capture accretion / 4th-order
Hermite sink integration, and the ideal / enforced-adiabat / barotropic equations of state.
No cooling, no MHD, no feedback, no cosmology, no restart files.

## Build

```bash
cd shmem
make GIZMO
```

Needs `mpicxx`, OpenMP and HDF5. On Flatiron machines: `module load modules gcc openmpi hdf5`
first (the Makefile locates HDF5 through `HDF5_BASE`, falling back to the nix-store path).

Two things people trip on:

* `-march=native`: build on the architecture you will run on. For cluster jobs, build on the
  node inside the job -- see the worktree pattern in the production sbatch scripts, and its
  warning: **a git worktree only sees commits**, so uncommitted changes are silently absent
  from such a build.
* The sources include `../math_types/*.h`, so the `shmem/` directory only builds from inside
  the repo (or a copy that preserves the parent layout).

## Run

```bash
GIZMO_CONFIG=/path/to/Config.sh ./GIZMO params.txt 0
```

Unlike the reference, the config is read at **runtime**, not compile time: one binary serves
every test and problem. At startup the engine prints any config flags it does not implement
and ignores them -- read that list before trusting a new setup. The params file is standard
GIZMO format; `DEVELOPER_MODE` in the config makes the engine honour the file's accuracy
parameters (CourantFac, ErrTolTheta, ...); without it the reference's built-in values are
substituted, exactly as real GIZMO does.

Threading and placement, for anything performance-sensitive:

```bash
export OMP_NUM_THREADS=<cores> OMP_PROC_BIND=spread OMP_PLACES=cores
numactl --interleave=all ./GIZMO params.txt 0
```

The interleave matters at production sizes: every bulk array is first-touched by one thread
during the IC read, and without it all of it lands on one NUMA domain. Do **not** wrap the
engine in `mpirun -np 1`: OpenMPI binds the process to one core at small np and every OpenMP
thread inherits that mask, silently serialising the run.

There is no restart machinery. Pointing `InitCondFile` at a snapshot continues the physical
state but restarts the clock at zero, so `TimeMax` and snapshot numbering must be rebased by
hand.

## Tests

The engine passes GIZMO's own pytest suite unmodified, via the prebuilt-binary hook:

```bash
GIZMO_PREBUILT=$PWD/shmem/GIZMO PYTHONPATH=$PWD/python_src python -m pytest test/<name>
```

The acceptance set this engine is developed against: shocktube, soundwave, square, gresho,
sedov, evrard, shu1977, plummer, hernquist, plummer_binaries. Run all of them for any change
to the neighbour search, the timestep machinery, or the step structure -- density feeds
everything, so "gravity-only" changes are rarely gravity-only. Known standing exceptions:
the plummer_binaries KDK variant fails its r_50 Lagrange-radius check (expected -- the KDK
integrator is not accurate enough for a binary-rich cluster; the Hermite variant passes),
and `suite_runner.cc` is bit-rotted.

## Runtime diagnostics (environment variables)

The engine carries its instrumentation as `SHMEM_*` environment variables; grep the sources
for the full set. The ones worth knowing:

| variable | effect |
|---|---|
| `SHMEM_PROFILE=1` | per-phase timers; first-8-step `[prof]` lines on stderr |
| `SHMEM_PROFILE_TOTALS=1` | cumulative `[prof-total]` + active-fraction `[prof-bucket]` every 100 steps -- the honest view of where a run's time goes |
| `SHMEM_SINKDT=1` | which criterion binds the sink timestep (`[sinkdt]`) |
| `SHMEM_BIN_DIAG=1` | why the deepest timebin is deep (h collapse vs signal speed) |
| `SHMEM_DT_DIAG=1` | one-shot dump of the global dt limiter |
| `SHMEM_MOMAUDIT=1` | per-phase momentum/angular-momentum bookkeeping probes |
| `SHMEM_ENERGY_LOG=1` | synchronised total energy per step. **Computes a full-box potential every step** -- built for 2-particle orbit tests; never enable on a production-size run |
| `SHMEM_GLOBAL_TIMESTEP=1` | disable individual timesteps (A/B against the hierarchy) |
| `SHMEM_DENSE_DRIFT=1` | restore the O(N) drift sweep (A/B against lazy drift) |
| `SHMEM_NO_HYDRO=1` | gravity-only free fall (separates tree asymmetry from hydro) |
| `SHMEM_NO_LEAF_ACCEPT=1` | force-open tree leaves (multipole-acceptance A/B; slow, accurate) |

## Layout and further reading

* `mfm.cc` -- the step (`mfm_step`), density/gradients/fluxes, sinks, Hermite, timebins
* `hydro.cc` / `tree.cc` -- neighbour search and the tree (build, walks, node bounds)
* `gizmo_main.cc` -- driver: config/params parsing, IC load, snapshot output
* `PORTING_GOTCHAS.md` -- traps discovered while matching the reference; read before "fixing"
  anything that disagrees with real GIZMO
* `STEP_ORDER_RESTRUCTURE.md` -- why the step runs in GIZMO's exact phase order, the Pred-state
  machinery, and the measurements behind both

The prime directive when extending this engine: **use the reference implementation**. Every
substantive deviation that looked like an improvement has, on measurement, been a bug.
