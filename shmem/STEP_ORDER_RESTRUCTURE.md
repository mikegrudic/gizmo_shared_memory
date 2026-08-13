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

Branch `hermite_step_order`, forked from omp_shmem at 40dfdcbf. THE ROTATION IS IN: mfm_step now
runs GIZMO's phasing exactly --

    top:   gather actives(t)  ->  dt/bins from the PREVIOUS evaluation  ->  hermite snapshot
           ->  fused kick (kick#2 of the ended step + kick#1 of the new one, same rates)
    drift: everyone to t', t_since_build += dt, CLOCK ADVANCES (find_next_sync_point_and_drift)
    tail:  gather actives(t') -- a DIFFERENT set -- ->  evaluate_forces at the drifted positions
           ->  sinks  ->  hermite predict / pass#2 / correct  ->  node-bound top-up

Consequences that landed with it, all reference-mandated:

  * HYDRO RATES GO THROUGH THE KICKS (kicks.cc: HydroAccel and DtInternalEnergy times
    dt_hydrokick). The all-at-once conserved update is gone; fluxes() output is converted to
    a_hydro + du_dt per particle (hydro_final_operations_and_cleanup) and applied trapezoidally
    by the same pending-half-kick machinery gravity uses. The internal energy moves LINEARLY by
    its rate -- the reference linearises the same way, not an exact conserved-quantity solve.
  * predict_half is GONE. The reference's face reconstruction is spatial only
    (hydro_core_meshless.h); time centring comes from the two rate evaluations at the interval's
    ends. set_predicted_states publishes current primitives, and the kick refreshes them.
  * The position time-centring correction is GONE: the drift now uses the half-kicked velocity,
    which is the leapfrog's own time centring.
  * hermite_pass takes h from clock - herm_tick, which after the rotation IS the particle's own
    completed step (the clock advances before the pass). This also removed a real bug: Phase A
    stamped last_drift with the pre-advance clock while writing post-drift positions, so every
    corrected sink was over-drifted by one system step on its next touch -- candidate for the
    13x prefactor regression measured on the partial restructure.
  * Bootstrap: sim.forces_valid gates one evaluation at t=0 (GIZMO's init), so the first dt
    decision has forces.

suite_runner.cc was already broken before this work (stale density() call) and stays broken.

## Measured results

Binary (e=0.9, 10 orbits, scratchpad/binary/sweep.sh): Hermite error per halving of dt falls
46-110x => order ~5-6, matching the methods paper's "fifth order with these time-step criteria".
At eta=0.00125: 6.9e-9/orbit, below the reference's documented 1e-6. KDK bit-identical to its
pre-rotation values (the control). The old 13x prefactor and the order-2 ceiling are both gone;
the prefactor was Phase A stamping last_drift with the pre-advance clock, over-drifting every
corrected sink by one system step.

Soundwave convergence vs N (1D, one crossing, L1 against the IC, scratchpad/swconv/):

    N       ours rho    ref rho     ours u      ref u       ours v      ref v
    512     1.58e-4     7.66e-5     1.59e-4     7.66e-5     4.36e-4     4.30e-4
    2048    1.65e-5     5.03e-6     1.68e-5     5.86e-6     1.18e-4     1.18e-4

rho and u converge at ~2nd order in both codes (pre-rotation, rho was FIRST order). v shows
slope 1 in BOTH -- that is the snapshot kick-phase artifact, not integration error: io.cc:230
writes the raw half-kicked velocity on purpose, and our L1v matches the reference's to four
digits. Three Pred-machinery pieces were needed to get here, all reference-mandated:
  * set_predicted_states completes the owed half-kick with the stored rates (= VelPred);
    publishing the raw stored state instead measured 83x worse on the soundwave return.
  * gradients read the predicted fields on both sides, like every other hydro input.
  * snapshots write the PREDICTED internal energy (io.cc:276) while velocity stays raw.

OPEN: our rho/u error coefficient is ~3x the reference's at N=2048 (same order). Localised to
the h-solve's FIXED POINT, not the step scheme: with MaxNumNgbDeviation loosened to 0.05 we
reproduce the reference's coefficient (5.2e-6 vs its 5.0e-6), while tightening to 1e-10 makes it
WORSE (3.0e-5) -- yet the reference runs the same 1e-6 the suite sets and gets 5e-6. So the
neighbour-number functional our iteration converges to deviates from the reference's
(density.cc's kernel-weighted target + dhsml/Omega corrections are the place to diff). The
harder we converge onto our functional, the further we land from the reference's answer.
