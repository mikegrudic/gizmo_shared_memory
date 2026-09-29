"""Pytest configuration: ensure tests import `gizmo` from the local
`python_src/` source tree, regardless of whether the package is pip-installed, and
guarantee each test leaves the working directory where it found it.

NOT enforced, but worth knowing: do not run two sessions against one working tree. Every
test builds ./GIZMO and writes Config.sh in the repo root, and writes test/<name>/output/
before moving it to a per-variant name, so concurrent sessions silently interleave
snapshots and swap each other's binaries. The symptoms are a variant directory whose
snapshot mtimes are not monotonic, a run reporting no snapshots at all because another
session moved output/ out from under it, or a variant whose startup config listing does
not match the flags it was supposed to build with. Note the sbatch wrapper does NOT create
a worktree despite its name; it cds to $SLURM_SUBMIT_DIR."""

import os
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SRC = REPO_ROOT / "python_src"

if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))


# Config.sh flags the shared-memory engine in shmem/ does not implement. A variant that turns one
# of these on does not exercise the feature -- the engine's config parser simply ignores the flag
# and runs the baseline -- so the run is wall-clock spent to reach a foregone failure against a
# ceiling calibrated for the feature. Worse, a permanently-red test is a test nobody reads: five of
# these were failing on every run, and one of them (RANDOMIZE_GRAVTREE, since implemented) was correctly reporting a
# real deficiency that went unnoticed for exactly that reason.
#
# Skipped only under GIZMO_PREBUILT; the full GIZMO build implements all of these and must still be
# held to them. Delete an entry the moment the engine gains the feature -- the skip reason names it.
PREBUILT_UNIMPLEMENTED = {
    "PMGRID": "no particle-mesh long-range gravity",
    "BOX_PERIODIC": "no Ewald summation for periodic gravity",
    "ADAPTIVE_GRAVSOFT_FORALL": "adaptive softening is gas-only (ADAPTIVE_GRAVSOFT_FORGAS)",
}


@pytest.fixture(autouse=True)
def skip_unimplemented_variants(request):
    """Skip variants whose Config.sh flags the prebuilt engine does not implement.

    Keyed on the `extra_config_flags` parameter the variant tests already parametrise over, so it
    costs nothing at the call sites. A flag matches if it appears as the whole token or as the
    `NAME=value` form (`PMGRID=64`).
    """
    if not os.environ.get("GIZMO_PREBUILT"):
        return
    flags = request.node.callspec.params.get("extra_config_flags", ()) \
        if hasattr(request.node, "callspec") else ()
    for flag in flags or ():
        name = str(flag).split("=", 1)[0].strip()
        if name in PREBUILT_UNIMPLEMENTED:
            pytest.skip(f"{name}: {PREBUILT_UNIMPLEMENTED[name]} in the shmem engine "
                        f"(GIZMO_PREBUILT); variant would run as baseline")


@pytest.fixture(autouse=True)
def restore_cwd():
    """Restore the working directory after every test, however the test exits.

    Many tests chdir into test/<name>/ and chdir back with a trailing `chdir("../../")`
    that is NOT in a finally block, so anything raising in between leaks the cwd to every
    subsequent test -- and the expected path does raise, since run_test() implements the
    GIZMO timeout as pytest.skip(). Observed: one c_shock timeout left the cwd there, so
    every later `make clean` ran from the wrong directory, giving 54 spurious build
    failures. A safety net, not a licence to skip try/finally: it warns so the offending
    test still gets fixed.
    """
    original = os.getcwd()
    try:
        yield
    finally:
        current = os.getcwd()
        if current != original:
            os.chdir(original)
            warnings.warn(
                f"test left cwd at {current!r} instead of {original!r}; restored. "
                "Wrap the chdir in try/finally (or use build_and_run_test, which does).",
                stacklevel=1,
            )
