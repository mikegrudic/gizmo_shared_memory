# Building an MFM + gravity + sinks code: what to look out for

Written from porting GIZMO's MFM/STARFORGE physics into this shared-memory engine. Most of what
follows is invisible from the equations — it is only findable by reading the reference source or by
measuring against it. Every entry says **what it cost and what it bought**, including the several
that bought nothing, because knowing which leads are dead is worth as much as knowing which worked.

The failures share a shape: **the run keeps producing plausible output.** Almost nothing here
crashes. A wrong face weight gives a converged-looking collapse that fragments; a sub-tick clock
advance gives smooth output with gravity switched off; a stale density gives a shock with a 25%
pre-shock deficit. Assume silent wrongness, and measure against a reference rather than eyeballing.

---

## 0. The meta-lesson

**When a ported feature misbehaves, read the reference before theorising.** Twice in one day this
cost hours: sink formation produced no sinks and the response was reasoning about whether a
time-averaged virial criterion could converge, instead of diffing against `galaxy_sf/sfr_eff.cc`;
and a lazy-drift scheme designed from first principles (per-particle last-drift tick, global speed
bound for the tree pad, halo catch-up pre-pass) came out **2× slower** — while GIZMO already
implements lazy drift in twelve lines, and two of the three invented mechanisms *were* the
regression.

The corollary that matters for debugging: when your code and the reference disagree, the bug is
much more often in a definition you assumed than in the algorithm you implemented.

---

## 1. MFM hydro

### Face weights must be centred when volumes differ (`compute_finitevol_faces.h:15-18`)

`A_ij = wk_i·wt_i·B_i·dx + wk_j·wt_j·B_j·dx`. The Lanson & Vila default `wt = V` **assumes h barely
varies between neighbours** — false everywhere in a collapse. GIZMO switches both sides to one
centred weight when `|V_i−V_j|/min(V)/NUMDIMS > 1.25`:

```c
wt_i = wt_j = V_i*V_j*(wk_i+wk_j)/(V_i*wk_i + V_j*wk_j);
```

The face stays antisymmetric either way, so **linear momentum is conserved and nothing looks
wrong** — but the volume gradient tilts the face normal, and the tangential residual torques
infalling gas.

**Cost:** the single largest bug found. **Bought:** shu1977 went from fragmenting into 3 sinks to
10/10 runs with exactly one sink, and four separate symptoms resolved at once (they were one bug):

| at snapshot 4 | reference | broken | fixed |
|---|---|---|---|
| Msink | 0.0376 | 0.0275 | 0.0366–0.0370 |
| j/j_circ (2–4 / 4–8 / 8–16 r_s) | 0.164 / 0.100 / 0.124 | 0.892 / 0.931 / 0.542 | 0.07–0.24 / 0.22–0.38 / 0.17–0.19 |
| N(2–16 r_s) | 21 | 65 | 11–16 |
| \|cusp − sink\| / r_s | 0.29 | 0.80 | 0.20–0.95 |

Torqued gas cannot fall in → piles up → is not accreted (Msink low) → goes self-gravitating →
fragments.

### Antisymmetric faces conserve momentum but NOT angular momentum

`A_ij = −A_ji` holds by construction, so every pair conserves linear momentum exactly. But the force
acts along `Â`, which is **not** parallel to the separation, so `r_i×F_i + r_j×F_j ≠ 0`. Angular
momentum is only ever approximately conserved in MFM, and any systematic bias in the face direction
accumulates as spin-up. Watch `j/j_circ`, not momentum, when chasing rotation.

### Degenerate geometry needs a radial fallback (`compute_finitevol_faces.h:56-60`)

When the condition number blows up, or `A·dp < 0` (impossible for a positive-definite `E`, so it
signals numerical degeneracy), GIZMO replaces the face with one **along the separation** — exactly
central, therefore torque-free for that pair. Without it, degenerate cells emit tilted faces.

### Clamp the HLLC contact speed into the wave fan (`reimann.h:572`)

`S_M = num/den` with only `|den| > 1e-300` as a guard returns absurd values for near-degenerate
states — measured **1e5 against a sound speed of 0.2**. `S_M` also sets the face energy flux, so
this is wrong independently of any timestep use.

### There are several "signal velocities" and they are not interchangeable

For MFM+GALSF, GIZMO *replaces* the Monaghan estimate with the contact wave speed
(`hydro_core_meshless.h:253`): `vsig = 2·S_M + max(0, Δv_face)`. Two traps:
- The sign convention. GIZMO forms `vdotr2 = (x_i−x_j)·(v_i−v_j)` and boosts `vsig` when it is
  **negative** (closing). Getting this backwards loses the velocity term at exactly the shock fronts
  and collapse regions that need it — and it will still pass most tests.
- **Whatever you store must be what you compare against.** Storing the contact-wave `vsig` while the
  wakeup test computes a Monaghan pair speed makes the ratio ~40 against a threshold of 4.1, so
  every inactive neighbour is demoted three bins every step. The box runs away to bin 21 while every
  `desired_dt` criterion still reads large (`cfl = 0.28` on cells sitting in the deepest bin).

### Reconstruction details worth checking against source

Face position (`s_star_ij`, and note both off-centre formulas are commented out in the reference —
it *is* the midpoint), `v_frame = ½(v_i+v_j)`, limiter constants (`fac_minmax=0.5, fac_meddev=0.375`
at `SLOPE_LIMITER_TOLERANCE=1`, but 0.75/0.40 at 2), and the flux de-boost `E += v_frame·F_v`. All
of these matched here; they are listed because each is a plausible place to differ silently.

---

## 2. Individual timesteps — the richest source of silent wrongness

### Accumulate RATES in the pair loop; never touch an inactive neighbour

The tempting design — time-integrate the flux inside the pair loop and apply it to both endpoints —
buys exact cross-bin conservation and then silently obliges the code to know **how long each pair
has been integrated**. That is not derivable from the bins, because the wakeup deliberately changes
a particle's bin mid-step. Result: flux applied over wrong intervals, spurious heating, timebins
driven ever deeper.

GIZMO instead accumulates rates with no `dt` anywhere in the loop, updates `j` only if it is active
(and `j_is_active_for_fluxes` is hardcoded 0, so in practice never), and lets each particle
integrate its own rate over its own step at kick time. Conservation is exact between binmates and
approximate across a bin boundary — the deliberate trade that removes all pair time-accounting.

**Bought:** fixed `test/square` outright (pressure spread exactly 1.00000 as required by
construction, bins pinned at 5–6, matching the global-timestep run).

### The bin-assignment ratchet

Find an aligned bin by walking **up from the particle's current bin** (aligned by construction,
since it is active), never by walking deeper from the desired bin. The latter is a ratchet: once
anything reaches a deep bin the clock only advances in that bin's ticks, so a particle wanting a
shallow bin finds it unaligned on most syncs and is pushed deep purely for alignment. The whole box
migrates to the deepest bin and can never climb back — running 20× the syncs it needs while every
physical quantity looks perfectly healthy.

### Every step must advance the integer clock by a whole number of ticks

A particle is active iff `clock_ticks % ticks_in_bin(bin) == 0`. Bins are powers of two, so
alignment is all-or-nothing: advance the clock by a non-tick-multiple **once** and nothing is ever
aligned again — the active set is empty on every subsequent sync, forever.

That failure is silent in the worst way. With an empty active set nothing is kicked, but the drift
pass still moves everyone and `deepest_active` falls back to 0, so `dt = dt_base` and the run
continues at a plausible step size producing plausible output — **with gravity effectively switched
off**. Velocities frozen, positions advancing.

How it bit: the driver accumulated `time += dt` in floating point and closed snapshot boundaries
with `min(dt_max, target − time)`. After nine intervals the FP residual exceeded the write
tolerance, one extra step ran with `dt ~ 5e-15`, `llround(dt/dt_base·2^30)` gave 0, a
`if (ticks < 1) ticks = 1` clamp bumped it to 1, and the final snapshot interval (~47 steps) ran
ballistically: energy −0.146 → +0.001, Lagrange radii +190%, **KE identical to five digits**.

Fix: make the integer clock the sole source of truth, quantise the caller's cap down to whole ticks,
and never clamp a sub-tick request up to 1.

### Inactive particles need drift-time density prediction

GIZMO predicts `Density *= exp(−divv·dt_drift)` for inactive particles at drift time
(`core/predict.cc:206`, `divv_fac` clamped to ±0.3). Without it, long-binned inflow carries a stale
density — noh showed a **25% pre-shock deficit** with `vr = −1` exactly, which reads as a hydro bug
and is not one.

### The wakeup is an encoding, not a bin

`P[j].wakeup = local.TimeBin + 1` (0 means "no request"), decoded at `timestep.cc:1350` as
`bin = waker_bin − ceil(log2(WAKEUP))`. The woken cell lands on a step **8× shorter than the
waker's**, not one bin longer. Read as a bin assignment, the limiter inverts and weakens.

---

## 3. Gravity and the tree

### Accuracy and symmetry are different axes — optimise the right one

Measured against an exact O(N²) sum on identical positions, at matched theta and group size:

| solver | \|Δa\|/\|a\| median | \|torque\| | torque per unit error |
|---|---|---|---|
| exact O(N²) | 0 | 1.9e-15 | — |
| this tree | 4.6e-04 | 7.9e-06 | 1.7e-02 |
| GIZMO | 9.0e-04 | 5.0e-06 | 5.6e-03 |

GIZMO is **2× less accurate** and injects **1.6× less spurious torque**. Its per-particle
contributions are 2.2× *larger*; they simply cancel (coherence 0.008 vs 0.026). `torque =
coherence × Σ|L|` reproduces both. Chasing force accuracy is chasing the wrong number.

### Tree torque is grid-imprinted

Shifting the root box by 1.3% of a side leaves `|Δa|` essentially unchanged (3.7–4.9e-4 across six
placements) but **completely reorients the torque** each time; averaging over six placements
suppresses the net 9×. That is the mechanism `RANDOMIZE_GRAVTREE` exploits. **But** — it bought
GIZMO only **1.19×** on shu1977's momentum drift (vs 23–34× on plummer/hernquist), so the mechanism
being real does not make it a fix for your problem. Check the reference's own calibration numbers
before porting.

### Drift on touch, inside the walk, before the distance test

Not a halo pre-pass, not an over-inclusive search radius, not a global speed bound on the tree pad —
all three were invented here and two were a measured 2× regression. GIZMO drifts a particle the
moment a search reaches it (`forcetree.cc:1678`, and the ngb codeblocks), **before** the distance
test, so a stale position can never hide a true neighbour. The slack lives in the node extent.

### Batching is not grouping

Chunking a Morton-sorted target list into fixed windows and taking each window's bounding box lets a
window **straddle a high-level Morton boundary**, where adjacent indices are spatially far apart —
the bbox explodes and every member pays for the over-opening. Measured **+76%**. Using *tree nodes*
as the target groups makes the extent tight by construction; that (and dual-tree traversal) is the
structure worth building, and remains untested here.

### Opening criteria are config-gated

`GRAVITY_HYBRID_OPENING_CRIT` (BH ∪ relative) exists only under
`GRAVITY_ACCURATE_FEWBODY_INTEGRATION`, inside the STARFORGE block. Everywhere else
`gravtree.cc:489` zeroes `ErrTolTheta` after the first walk and the relative test runs alone.
Applying the union unconditionally cost evrard ~34% of its per-step time computing a force the
reference does not compute. Also note the relative test needs `aold`, which does not exist on step
one — hence GIZMO's throwaway `gravity_tree()` at `Ti_Current == 0` (`accel.cc:49`), and its
skipping of the softening/relative block there. Scoring a walk at `Ti_Current == 0` therefore
measures Barnes-Hut alone, not the production criterion.

### Adaptive softening: symmetrise, and pair the zeta terms with the averaged kernel

For gas–gas pairs the kernel is the **average** of the two softened kernels,
`0.5·(g(r;h_i) + g(r;h_j))`, not the kernel of `max(h)`. The Price & Monaghan zeta corrections are
*derived* for the averaged form — pairing them with a max-softening kernel mis-cancels. Divide the
correction by the **target's** mass, which is what keeps the pair antisymmetric
(`m_i a_i = −m_j a_j`). Zeta is gas–gas only; nodes get max-softening and no zeta.

Related trap: evrard's central-density deficit looked like a zeta problem and was actually the
**slope limiters** (`a_limiter` reach and face overshoot). Entropy was the tell.

---

## 4. Sinks

- **Formation criteria are vetoes applied in a specific ORDER**, on a rate then multiplied by 1e20 —
  a cell that survives every test converts deterministically. Getting the order (or the `tsfr` units)
  wrong yields *no sinks at all*, which reads as a physics problem and is a porting problem.
- **Only inflow is excused from the virial criterion**, matching the reference.
- **Accretion has five gates**, and they are easy to get subtly wrong: boundedness `(v²+cs²)/vesc² < 1`
  with `cs² = 3P/ρ` under the γ≈1 isothermal hack (not the actual sound speed); `vesc` spline-softened
  and including an isothermal-sphere gas term; Bate's `j² < G(M+m)·r_sink`; a resolution cut
  `psize > 1.396263·r_sink`; and a `SwallowTime` tie-break that only matters with multiple sinks.
- **`SINK_ALPHADISK_ACCRETION` is a STARFORGE default**, so swallowed gas enters a reservoir and
  drains over `t_acc` rather than becoming stellar mass on contact. Under
  `SINK_GRAVCAPTURE_FIXEDSINKRADIUS` that collapses to a constant fixed at formation,
  `t_acc = G·M_form/cs_min³` — i.e. `mdot = (cs_min³/G)·(reservoir/M_form)`, the Shu isothermal rate
  scaled by how full the disk is, with `cs_min` a hard 0.2 km/s floor. This drives `dt_accr`, which
  binds ~40% of the time early on.
- **Accretion can empty the active bins.** Removing a particle can empty the bins that alone were
  aligned with the current clock (the swallowed cells near a sink are exactly the deepest-binned
  ones), leaving a tick no survivor syncs on. Fast-forward the clock to the earliest tick any
  survivor is aligned to.
- **Deleting a particle invalidates every index the tree and neighbour cache hold.** Scan and decide
  first, touching nothing; apply deletions afterwards, descending. Any per-particle array not
  included in the swap/pop bookkeeping silently desyncs.
- **The accretion radius floors on the sink's force-softening kernel radius**, not the progenitor
  cell's smoothing length (`sfr_eff.cc:608`). Getting that wrong made it 1.78× the reference's.
- **`vesc` is spline-softened** (`sink.cc:103`), not bare `1/r` — the latter is far too permissive
  inside the softening. The isothermal-sphere `m_eff` term is gated on `SINGLE_STAR_SINK_DYNAMICS`,
  *not* on `COOLING`, so it is always on for a STARFORGE run.
- **Watch what else is gated on `COOLING`.** The opacity-limited `cs` scaling and the 0.1 AU
  Larson-core floor are gated on `COOLING || EOS_GMC_BAROTROPIC` — neither of which an
  `EOS_ENFORCE_ADIABAT` run has. Applying them unconditionally quietly altered that test.

### Collisional dynamics (Hermite, binaries)

- **Bin conventions are inverted between codes, and getting it backwards is a silent no-op.** GIZMO
  uses `dt = 2^bin`, so *higher bin = longer step*; this engine counts the opposite way. GIZMO's
  **minimum** TimeBin over the kernel gas is therefore this engine's **maximum** bin. Inverted, the
  sink-gas cap keys on the *longest*-stepping neighbour and does nothing at all — no error, no
  symptom, just an absent constraint.
- **Take the Hermite interval from the integer clock, not a stored `dt`** (`kicks.cc:134-138`,
  `tstart = Ti_begstep`). Storing the assigned `dt` is wrong whenever the step *taken* differs from
  the step *assigned* — and that is routine: every snapshot boundary truncates `dt` via the cap, and
  any Saitoh-Makino wakeup deepens a bin mid-step.
- **Phase the pass: predict all → evaluate all → correct all, with barriers.** A fused per-target
  loop races on binary partners — one thread reads a partner mid-overwrite. That cost a **factor of
  ~10 in energy error** before it was found, and a binary is exactly the configuration that exposes
  it, because the two members are each other's dominant force.
- **Keep KDK running underneath** and overwrite it for eligible particles only. Newborn sinks and any
  sink that accreted this step must fall back (`kicks.cc:104`), so the base integrator has to be
  valid at all times rather than replaced.
- **Compute the jerk inside the existing gravity walk** (`forcetree.cc:2266`), reusing the traversal
  and the kernel derivatives the tidal tensor already needs — a second walk for it is pure waste.
- **`dt_2body` is the harmonic mean** of the approach and freefall times to the nearest sink, taken
  from per-particle minima filled during the gravity walk (`forcetree.cc:2509`).
- **Give the test a KDK variant.** `plummer_binaries` parametrises `DISABLE_HERMITE_INTEGRATION`
  against the default, which makes it a true A/B of the integrator on identical ICs — the only clean
  way to attribute an energy-error change to Hermite rather than to everything else.

---

## 5. Configuration and harness traps

### The reference may not run its own parameter file

Without `DEVELOPER_MODE`, GIZMO **discards** `CourantFac`, `ErrTolIntAccuracy`, `ErrTolTheta` and
`ErrTolForceAcc` from the paramfile and substitutes compiled-in defaults, warning `Tag ... was
specified, but it is being ignored` (`begrun.cc:2637-2677`), then tightens them under
`GRAVITY_ACCURATE_FEWBODY_INTEGRATION`. shu1977 asks for `CourantFac 0.2` / `ErrTolTheta 0.21`; the
reference actually runs **0.4** and **0.5**. Reading the file instead put every Courant step at half
the reference's — and explains why tuning `ErrTolTheta` had no effect for hours: the reference was
never using the value being tuned.

### Expression-level traps — read the lines *around* the formula

- `Get_Particle_Size() = 1.61199·h/NumNgb` looks wrong until you find `density.cc:1064`, which
  overwrites `NumNgb` with its `NUMDIMS`-th root to save `cbrt` calls. It is `V^⅓`.
- `FaceClosureError` uses `dx_i = sqrt(V·trace(E))`, not `V^(1/dim)` — `density.cc:545` overwrites
  the obvious definition.
- Under `TIDAL_TIMESTEP_CRITERION`, the acceleration criterion sees **only** `HydroAccel`; the
  gravitational part is zeroed (`timestep.cc:357-379`) because the tidal tensor already supplies the
  gravity timestep. Counting it twice pinned 98.5% of cells to their softening-over-gravity time.

### The test infrastructure can be the bug

`test/square` failed with `LinAlgError: Matrix is singular` from three frames deep inside meshoid —
`Slice(order=1)` inverts a 3×3 gradient matrix, and the test is strictly 2D (all z = 0). Verified by
running the same call on the pristine ICs with no simulation involved. Reads as "the engine broke
square" while the physics assertions were passing all along.

Also: a permanently-red test teaches you to ignore the suite. Five variants here fail on every run
because the engine implements no `RANDOMIZE_GRAVTREE`/PM/Ewald — and one of those failures was
*correctly reporting* the grid-imprinted-torque deficiency. Either implement or skip, with the reason
recorded; do not leave standing red.

---

## 6. Diagnostics worth building early

Env-gated, no-op by default. The two that repeatedly earned their keep:

- **Disable a whole subsystem.** `SHMEM_NO_HYDRO` (pure gravitational free-fall) localised a
  day-long investigation in a single run: gravity alone reproduced the reference's symmetry
  (j/jc 0.167 vs 0.116) and passed with one sink, so the spin-up had to be hydro.
- **Score the solver against ground truth, reporting error AND torque separately.** They are
  different quantities and a solver can be better on one and worse on the other; conflating them sent
  this port after force accuracy for hours. This one overturned three separate conclusions.

Others that paid off: exact O(N²) gravity as a drop-in, a root-box shift to test grid imprinting,
per-particle timestep-criterion dumps on both codes for ID-matched comparison, and matching probes in
the *reference* so both sides can be measured on identical positions.

---

## 7. What each test can expose — and what it structurally cannot

The single most useful scheduling fact: **a passing test usually cannot see the bug the next test
will catch.** Several defects here survived every test that was green at the time, not by luck but
because those tests exercise the wrong regime.

| test | catches | blind to |
|---|---|---|
| soundwave | integrator order, phase error | anything nonlinear |
| shocktube / Sod | shock dissipation, limiter *safety* | over-limiting in smooth flow (clipping is *wanted* here) |
| square | Galilean invariance, periodic wrapping | anything with a density gradient |
| gresho | over-restrictive limiting, angular-momentum bleed | shocks |
| noh | stale state on **inactive** particles | anything where most particles are active |
| evrard | shock dissipation vs force error (via entropy) | individual-timestep staleness (nearly all active) |
| plummer / hernquist | collisionless dynamics, COM drift, correlated force error | hydro entirely |
| shu1977 | MFM faces in a steep density gradient, sinks | smooth flow |

### The specific lessons

**Per-face slope limiting is far more restrictive than it looks (gresho).** Limiting the *slope* on
every face means any neighbouring pair that happens to have a small jump clamps the gradient — even
where the field is exactly linear — so smooth extrema get clipped every step. The clipped slope
bleeds angular momentum continuously and the vortex spins down (v_phi 0.60 against an analytic 1.0,
L1 = 0.28 against a 0.15 tolerance). **A shock tube cannot reveal this**, because clipping is the
desired behaviour there. Fix: limit per particle, once, against the whole neighbourhood (Hopkins
2015 appendix B), with the face check demoted to a safety net.

**Use entropy to separate dissipation errors from force errors (evrard).** Central compression sat
at 0.85 of the reference with every assertion passing. Four faithfully-ported hypotheses moved it
essentially not at all (zeta corrections → 0.848, exact dt criteria → 0.850, opening criterion →
0.851). The tell was that core `s = P/ρ^γ` ran **3–11% high** while energy and momentum conserved to
round-off — which indicts shock dissipation and exonerates the force solver. The answer was in the
hydro: the gradient limiter's reach was 0.5h against GIZMO's `a_limiter = 0.25` (half the allowed
slope), plus a missing entropic-EOS face correction. 0.85 → 1.34 → 1.00–1.04.

**MFM is Lagrangian, so the timestep must be Galilean invariant (square).** A CFL built on the
lab-frame speed `c + |v|` collapses `dt` under a uniform boost. The square ICs advect at
`vx = 1243, vy = −358`, giving `dt` 865× too small and ~8.5M steps to reach TimeMax. The signal
speed must be built from **relative** velocities only. Second bug in the same test: position folding
assumed a drift never crosses more than one box length — at the corrected `dt` the square advects
1.32 box lengths per step, coordinates escaped the box, and it surfaced inside scipy's KDTree rather
than anywhere in the physics.

**Stale state on inactive particles needs a test where most particles are inactive (noh).** Under
individual timesteps a particle's density is frozen between its own updates, but in converging flow
the true density keeps rising. Noh's cold supersonic infall sat at **0.70** of the analytic
pre-shock profile with velocities exact to 4 decimals and `u` at the floor — only the density
estimate was stale. Evrard structurally could not show this (nearly everything active every step).
Fix: predict `ρ *= exp(−div v · dt)` at drift time, clamped to ±0.3, with `h` following the opposite
way. 0.70 → 0.967/0.983/0.992.

**Look at the profile, not the error norm (soundwave/Sod).** Plotting turned a scalar L1 into two
diagnoses it had hidden: a clean systematic *phase lag* with no damping identified the drift as
backward-Euler on position (time-centring it with `(v_old+v_new)/2` improved L1 by 3.4×), and the Sod
panels showed the dominant remaining defect was not diffusion at all but transverse velocity stripes
where a 2:1 lattice mismatch met the contact discontinuity — which the L1 number alone
mischaracterised.

**Periodicity is a compile flag, not `BoxSize > 0` (plummer).** Both plummer (300) and shu1977 (4.34)
carry a nonzero `BoxSize` through runs that are physically open. Keying the wrap on the parameter
silently makes open problems periodic.

**Correlated force error shows up as secular COM drift (plummer/hernquist).** The flag-off drift is a
lottery — it spans 30× between problems — so assert on a *ceiling* for the randomised path rather
than on a randomised-beats-flag-off *ratio*, or the test is flaky. And expect it to be
problem-dependent: randomisation buys 23–34× on plummer/hernquist and 1.19× on shu1977.

**Non-determinism is worth chasing before it hides something (MHD tests).** `mhd_blast` and
`orszag_tang` showed run-to-run varying divB errors and grid-aligned artefacts; the interim answer
was to force more of the work serial. Left as a known open item, but recorded because a
non-deterministic test cannot bisect a physics regression.

---

## 8. Dead ends, recorded so they are not re-run

| lead | verdict |
|---|---|
| Grouped tree walk correlating force errors | Refuted — `batch=1` gives *more* torque and *worse* accuracy than `batch=8`. |
| Stale node centres of mass (no `force_drift_node`) | Refuted — forcing a rebuild every step changed the gravity torque by **0%**. |
| `RANDOMIZE_GRAVTREE` decorrelation | Real mechanism (9× suppression measured), wrong problem — 1.19× on this test. |
| Tree force accuracy | Exonerated — exact gravity removes ~20% of the spin-up and still fragments. |
| Accretion criteria | Match line by line; the sink's cusp offset was a *symptom* of the hydro bug. |

**Four tests were themselves broken**, each producing a plausible result first: radialising gravity
about "the heaviest particle" (all gas starts at equal mass → arbitrary point → collapse suppressed,
0 sinks); radialising the total field about the collapse centre (destroys *local* self-gravity, first
sink 9× late behind a diffuse blob); radialising about the sink (which sits 0.8 r_s off the cusp,
leaving ~20% tangential force exactly where the null result was claimed); and scoring the reference's
walk at `Ti_Current == 0` (Barnes-Hut only, not the production criterion). Check that a test can
detect the effect it is looking for **before** believing its null result.
