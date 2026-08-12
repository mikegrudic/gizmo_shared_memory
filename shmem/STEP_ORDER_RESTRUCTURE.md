# Rotating mfm_step into GIZMO's order

Target (core/run.cc, per the reference's main loop):

    1  decide dt from the state left by the last step
    2  hermite-only gravity pass #1, save Old{Pos,Vel,Acc,Jerk}
    3  half-kick #1
    4  full-step drift
    5  density + gradients + gravity + hydro forces   <- AT THE DRIFTED POSITIONS
    6  half-kick #2
    7  hermite prediction
    8  hermite-only gravity pass #2
    9  hermite correction

Current mfm_step order:

    tree -> solve_h_and_volumes -> gradients -> compute_gravity -> sink_timestep_pass
         -> assign_bins / dt -> [hermite_snapshot, fused kick] -> flux -> drift
         -> hermite predict/pass#2/correct -> clock advance

## What already matches

Items 2, 4, 7, 8, 9 are in place (this session). Items 3 and 6 are equivalent as written: the
fused kick applies `a_grav * (pending + 0.5*dt)`, where the `pending` half IS the previous step's
kick #2 and uses `a(x(t_N))` -- the same acceleration GIZMO's kick #2 uses, since GIZMO evaluates
it post-drift at exactly those positions. Each acceleration serves kick #2 of one step and kick #1
of the next in both codes. Splitting them is relabelling, not a change of scheme.

## The actual work: item 5

Move the evaluation block (solve_h_and_volumes, gradients, compute_gravity, and the flux) from the
TOP of the step to AFTER drift_all_to. Consequences, all of which have to be handled together:

1. **dt comes from the previous step's state.** That is what the reference does ("decide timestep
   based upon state from last timestep"). The CFL term needs the signal speed from the previous
   gradients pass and the gravity term needs the previous a_grav. Both survive in sim; nothing new
   has to be stored, but assign_bins/dt must be moved ABOVE the evaluation block rather than below
   it.

2. **Bootstrap.** Step 0 has no previous evaluation. Needs one evaluation before the loop, or a
   first-step special case. The engine already does something similar for the initial density.

3. **The active set is still chosen at the top**, from the bins, exactly as GIZMO does
   (find_timesteps then make_list_of_active_particles). The post-drift evaluation is for THIS
   step's actives -- it feeds this step's kick #2 and the next step's dt.

4. **Flux ordering.** The flux loop currently runs pre-drift and writes wake requests that
   assign_bins consumes at the next sync. Moving it post-drift changes which sync sees a given
   wake request by one step. Check `sim.wake` handling.

5. **Profile timers** (t_dens/t_grad/t_grav/t_flux/t_drift) are read positionally; they move with
   their phases or the [prof] line silently mislabels.

6. **Sink passes.** sink_timestep_pass, sink_accel_check, sink formation and accretion all sit
   between the evaluation and the kick today. GIZMO runs calculate_non_standard_physics AFTER
   kick #2 and BEFORE the Hermite prediction (run.cc:230). Place them accordingly.

## Gate

The full ten-test acceptance set, not a subset: shocktube soundwave square gresho sedov evrard
shu1977 plummer hernquist plummer_binaries. This changes where every force is evaluated, so the
hydro tests are as exposed as the gravity ones. scratchpad/gate_full.sh runs them sequentially.

Plus the binary order test (scratchpad/binary/sweep.sh), which is the reason for the restructure:
Hermite currently measures order 2.85 in dt after this session's partial restructure, against the
methods paper's stated fifth order, and carries a ~13x prefactor regression that is not yet
explained.

## Status

Branch `hermite_step_order`, forked from omp_shmem at 40dfdcbf. Contains this session's Hermite
work (snapshot pass, predict/correct moved after the drift, pending-half-kick fix, h = dt_of,
SHMEM_ENERGY_LOG). Item 5 NOT started.
