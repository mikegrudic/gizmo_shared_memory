// GIZMO-compatible driver for the shared-memory MFM engine, so the repo's pytest suite runs
// UNMODIFIED against this code:
//
//   mpirun -np N ./GIZMO <paramsfile> <restartflag>
//
// MPI is initialised and every rank except 0 idles at a barrier: the whole point of this engine is
// that one shared-memory process does the work, but the harness launches through mpirun with
// whatever rank count the test parametrises, and that must Just Work. OpenMP threads come from
// OMP_NUM_THREADS exactly as the harness sets it.
//
// Reads: the params file (InitCondFile/OutputDir/TimeMax/TimeBetSnapshot/DesNumNgb/CourantFac/
// MaxSizeTimestep/BoxSize) and the Config.sh the harness copied to the repo root (for
// BOX_SPATIAL_DIMENSION and EOS_GAMMA -- compile-time constants in real GIZMO, runtime here).
// Writes: output/snapshot_NNN.hdf5 with the Header attrs and PartType0 fields the suite reads.

#include <sched.h>          // sched_setaffinity: undo mpirun's per-rank pinning (see main)
#include <unistd.h>

#include "mfm.h"
#include <hdf5.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace shmem;

static std::map<std::string, std::string> parse_kv(const char* path) {
    std::map<std::string, std::string> settings;
    FILE* file = fopen(path, "r");
    if (!file) return settings;
    char line[512];
    while (fgets(line, sizeof line, file)) {
        char key[128], value[256];
        if (line[0] == '%' || line[0] == '#') continue;
        if (sscanf(line, "%127s %255s", key, value) == 2) {
            // Values may be followed -- or entirely replaced -- by an inline comment, as
            // shu1977.params does for GravityConstantInternal ("%4301 ... calculated by code
            // if =0"). Treat a comment-leading value as absent rather than as atof()'s 0.
            if (value[0] != '%' && value[0] != '#') settings[key] = value;
        }
    }
    fclose(file);
    return settings;
}

// EOS_GAMMA=(5.0/3.0) / BOX_SPATIAL_DIMENSION=2 out of whichever Config.sh the harness staged
static void parse_config(int& n_dims, double& gamma, bool& gravity_on, bool& adaptive_soft,
                         bool& output_potential, bool& tidal_criterion, bool& box_periodic,
                         double& eos_adiabat, int& baro_variant, bool& baro_soundspeed,
                         bool& sink_formation, int& hermite_mask, bool& cooling_on,
                         bool& hybrid_opening, bool& developer_mode, bool& galsf,
                         bool& randomize_gravtree, double& atu_frac) {
    bool hermite_disabled = false;
    cooling_on = false;
    hybrid_opening = false;
    developer_mode = false;
    galsf = false;
    randomize_gravtree = false;
    atu_frac = 0.0;
    // Gravity is ON in GIZMO unless SELFGRAVITY_OFF is set, so default to on and let the config
    // switch it off -- the opposite default would silently drop gravity from any test whose
    // Config.sh simply does not mention it.
    gravity_on = true;
    adaptive_soft = false;
    output_potential = false;
    tidal_criterion = false;
    box_periodic = false;
    sink_formation = false;
    hermite_mask = 0;
    eos_adiabat = 0.0; baro_variant = -1; baro_soundspeed = false;
    // GIZMO_CONFIG (set by the pytest harness's GIZMO_PREBUILT path) names the staged config --
    // base + per-variant extra flags -- explicitly. The cwd/root fallbacks serve standalone runs;
    // under pytest the cwd copy is the pristine base WITHOUT the variant flags, which is exactly
    // why the explicit path must win when present.
    const char* env_config = getenv("GIZMO_CONFIG");
    const char* search_paths[3] = {env_config, "Config.sh", "../../Config.sh"};
    const char** paths_begin = env_config ? search_paths : search_paths + 1;
    const char** paths_end   = env_config ? search_paths + 1 : search_paths + 3;
    for (const char** p = paths_begin; p != paths_end; ++p) {
        const char* path = *p;
        FILE* file = fopen(path, "r");
        if (!file) continue;
        printf("shmem-GIZMO: config %s\n", path);
        std::vector<std::string> ignored;
        char line[512];
        while (fgets(line, sizeof line, file)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            bool known = false;
            auto flag = [&](const char* name) {
                const size_t n = strlen(name);
                // match the whole token, so OUTPUT_POTENTIAL does not also swallow a future
                // OUTPUT_POTENTIAL_FOO
                if (strncmp(line, name, n) != 0) return false;
                const char c = line[n];
                if (c != '\0' && c != '\n' && c != '=' && c != ' ' && c != '\t') return false;
                known = true;
                return true;
            };
            if (flag("SELFGRAVITY_OFF"))          gravity_on = false;
            if (flag("ADAPTIVE_GRAVSOFT_FORGAS")) adaptive_soft = true;
            if (flag("OUTPUT_POTENTIAL"))         output_potential = true;
            if (flag("TIDAL_TIMESTEP_CRITERION")) tidal_criterion = true;
            // Periodicity is a COMPILE flag in GIZMO, not a params entry: a params file may carry
            // BoxSize > 0 (plummer: 300, shu1977: 4.34) for a run that is nevertheless open.
            if (flag("BOX_PERIODIC"))             box_periodic = true;
            // The STARFORGE defaults switch on the whole single-star sink package; the engine
            // implements the formation criteria of that bundle (see sink_formation_pass).
            // The bundle also defines ADAPTIVE_GRAVSOFT_FORGAS (precompiler_logic.h:372) --
            // without it gas gravity is softened only by the fixed SofteningGas floor, which the
            // STARFORGE params set to ~0 precisely BECAUSE the softening is meant to be adaptive.
            // Missing this both unsoftens close gas-gas forces and drives the acceleration
            // timestep sqrt(eta*soft/|a|) orders of magnitude below the CFL step.
            // OUTPUT_POTENTIAL is also defined inside the STARFORGE defaults block
            // (precompiler_logic.h:361), and the tests read Potential from the snapshots.
            if (flag("SINGLE_STAR_SINK_FORMATION") || flag("SINGLE_STAR_STARFORGE_DEFAULTS") ||
                flag("SINGLE_STAR_SINK_DYNAMICS"))
                { sink_formation = true; tidal_criterion = true; adaptive_soft = true;
                  output_potential = true; }
            // The same bundle carries GRAVITY_ACCURATE_FEWBODY_INTEGRATION, hence the hybrid
            // opening criterion (see Sim::hybrid_opening). Everything else keeps the reference's
            // relative-only walk.
            if (flag("SINGLE_STAR_STARFORGE_DEFAULTS") ||
                flag("GRAVITY_ACCURATE_FEWBODY_INTEGRATION") ||
                flag("GRAVITY_HYBRID_OPENING_CRIT")) hybrid_opening = true;
            // GALSF, which the sink bundle defines rather than the config naming it directly:
            // SINGLE_STAR_STARFORGE_DEFAULTS -> SINGLE_STAR_SINK_DYNAMICS (precompiler_logic.h:368)
            // -> GALSF (:476). It gates the contact-wave signal velocity, which the reference
            // applies under MFM+GALSF only.
            if (flag("GALSF") || flag("SINGLE_STAR_STARFORGE_DEFAULTS") ||
                flag("SINGLE_STAR_SINK_DYNAMICS") || flag("SINGLE_STAR_SINK_FORMATION"))
                galsf = true;
            if (flag("RANDOMIZE_GRAVTREE")) randomize_gravtree = true;
            // ADAPTIVE_TREEFORCE_UPDATE=0.0625 is the reference's own default when the flag is
            // present without a value (precompiler_logic.h:381).
            double atu_value;
            if (flag("ADAPTIVE_TREEFORCE_UPDATE")) {
                atu_frac = (sscanf(line, "ADAPTIVE_TREEFORCE_UPDATE=%lf", &atu_value) == 1)
                         ? atu_value : 0.0625;
                // ATU sets its refresh cadence from the TIDAL timestep, so it needs that computed
                // whether or not the config asked for the tidal criterion itself. The reference
                // makes the same implication (precompiler_logic.h:600-603, "need this to estimate
                // the dynamical time"). Without it tdyn_for_treeforce stays 0, every particle is
                // forced fresh, and the whole feature is silently inert.
                tidal_criterion = true;
            }
            // Whether the paramfile's accuracy settings are honoured at all (begrun.cc:2637).
            if (flag("DEVELOPER_MODE")) developer_mode = true;
            // HERMITE_INTEGRATION 32 rides in the STARFORGE bundle (precompiler_logic.h:369)
            // unless PMGRID or DISABLE_HERMITE_INTEGRATION -- the latter is exactly what the
            // plummer_binaries "kdk" pytest variant appends, so the variants become a true
            // KDK-vs-Hermite A/B of this engine.
            if (flag("SINGLE_STAR_STARFORGE_DEFAULTS") && hermite_mask == 0) hermite_mask = 32;
            int hermite_value;
            if (flag("HERMITE_INTEGRATION") &&
                sscanf(line, "HERMITE_INTEGRATION=%d", &hermite_value) == 1)
                hermite_mask = hermite_value;
            if (flag("DISABLE_HERMITE_INTEGRATION") || flag("PMGRID")) hermite_disabled = true;
            if (flag("COOLING")) cooling_on = true;
            if (flag("EOS_GMC_BAROTROPIC_SOUNDSPEED")) baro_soundspeed = true;
            double adiabat_value;
            if (flag("EOS_ENFORCE_ADIABAT") &&
                sscanf(line, "EOS_ENFORCE_ADIABAT=%lf", &adiabat_value) == 1)
                eos_adiabat = adiabat_value;
            int baro_value;
            if (flag("EOS_GMC_BAROTROPIC"))
                baro_variant = (sscanf(line, "EOS_GMC_BAROTROPIC=%d", &baro_value) == 1)
                             ? baro_value : 0;   // bare flag = the MI2000 piecewise form
            int dims_from_config;
            if (flag("BOX_SPATIAL_DIMENSION") &&
                sscanf(line, "BOX_SPATIAL_DIMENSION=%d", &dims_from_config) == 1)
                n_dims = dims_from_config;
            double numerator, denominator;
            if (flag("EOS_GAMMA")) {
                if (sscanf(line, "EOS_GAMMA=(%lf/%lf)", &numerator, &denominator) == 2)
                    gamma = numerator / denominator;
                else if (sscanf(line, "EOS_GAMMA=(%lf)", &numerator) == 1) gamma = numerator;
                else if (sscanf(line, "EOS_GAMMA=%lf", &numerator) == 1) gamma = numerator;
            }
            // Flags this engine has no implementation for. Reported rather than ignored in
            // silence: a suite variant exists precisely to exercise the feature its flag names,
            // so running it as if the flag were absent makes the variant a duplicate of baseline
            // that PASSES -- which is how plummer/tidal and evrard/tidal_adaptive went green
            // without ever enabling the criterion they are named after.
            // ...except the ones this engine satisfies unconditionally, which would otherwise
            // make the warning pure noise: MFM is what the engine IS, output is always double,
            // and DEVELOPER_MODE only exposes extra params (already read by name).
            for (const char* benign : {"HYDRO_MESHLESS_FINITE_MASS", "OUTPUT_IN_DOUBLEPRECISION",
                                       "DEVELOPER_MODE"})
                flag(benign);
            if (!known) {
                std::string name(line);
                const size_t comment = name.find('#');       // "FLAG   # why" -- keep just FLAG
                if (comment != std::string::npos) name.resize(comment);
                while (!name.empty() && (name.back() == '\n' || name.back() == '\r' ||
                                         name.back() == ' ' || name.back() == '\t')) name.pop_back();
                if (!name.empty()) ignored.push_back(name);
            }
        }
        fclose(file);
        if (!ignored.empty()) {
            printf("shmem-GIZMO: WARNING -- %zu config flag(s) NOT implemented by this engine "
                   "and ignored:\n", ignored.size());
            for (const auto& name : ignored) printf("shmem-GIZMO:     %s\n", name.c_str());
        }
        break;                                   // first Config.sh found wins (cwd over root)
    }
    if (hermite_disabled) hermite_mask = 0;
}

// Read one dataset as doubles. `column` selects a component of an Nx3 dataset; pass -1 for a
// plain 1D dataset.
static std::vector<double> h5_read(hid_t group, const char* name, int column) {
    hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dataset);
    hsize_t dims[2] = {0, 0};
    const int rank = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    const size_t n_rows = dims[0];
    std::vector<double> values(n_rows);
    if (rank == 1) {
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    } else {
        std::vector<double> all_columns(n_rows * dims[1]);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, all_columns.data());
        for (size_t i = 0; i < n_rows; ++i)
            values[i] = all_columns[i*dims[1] + (column < 0 ? 0 : column)];
    }
    H5Sclose(space); H5Dclose(dataset);
    return values;
}

static void write_attr(hid_t where, const char* name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(where, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &value); H5Aclose(attr); H5Sclose(space);
}
static void write_attr(hid_t where, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(where, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &value); H5Aclose(attr); H5Sclose(space);
}
// the per-particle-type header arrays GIZMO writes, one entry per type
template <typename T>
static void write_attr_per_type(hid_t where, const char* name, hid_t type, const T* values) {
    hsize_t n_types = 6;
    hid_t space = H5Screate_simple(1, &n_types, nullptr);
    hid_t attr = H5Acreate2(where, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, values); H5Aclose(attr); H5Sclose(space);
}

static void write_snapshot(const Sim& sim, const std::vector<long long>& particle_ids,
                           const std::string& outdir, int snapshot_num, double time,
                           double header_box) {
    char filename[512];
    snprintf(filename, sizeof filename, "%s/snapshot_%03d.hdf5", outdir.c_str(), snapshot_num);
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    const size_t n_part = sim.size();

    // Types were loaded in ascending order, so each type is one contiguous run of indices.
    unsigned int count_per_type[6] = {0,0,0,0,0,0};
    if (sim.P.type.empty()) count_per_type[0] = (unsigned)n_part;
    else for (size_t i = 0; i < n_part; ++i) count_per_type[sim.P.type[i]]++;
    size_t type_start[6];
    for (int t = 0, at = 0; t < 6; ++t) { type_start[t] = at; at += count_per_type[t]; }

    hid_t header = H5Gcreate2(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    unsigned int zeros[6] = {0,0,0,0,0,0};
    int count_per_type_int[6];
    for (int t = 0; t < 6; ++t) count_per_type_int[t] = (int)count_per_type[t];
    double mass_table[6] = {0,0,0,0,0,0};       // 0 => per-particle masses live in the dataset
    write_attr_per_type(header, "NumPart_ThisFile", H5T_NATIVE_INT, count_per_type_int);
    write_attr_per_type(header, "NumPart_Total", H5T_NATIVE_UINT, count_per_type);
    write_attr_per_type(header, "NumPart_Total_HighWord", H5T_NATIVE_UINT, zeros);
    write_attr_per_type(header, "MassTable", H5T_NATIVE_DOUBLE, mass_table);
    write_attr(header, "Time", time);
    write_attr(header, "Redshift", 0.0);
    write_attr(header, "BoxSize", header_box);
    write_attr(header, "NumFilesPerSnapshot", 1);
    write_attr(header, "Flag_Sfr", 0);      write_attr(header, "Flag_Cooling", 0);
    write_attr(header, "Flag_Feedback", 0); write_attr(header, "Flag_StellarAge", 0);
    write_attr(header, "Flag_Metals", 0);   write_attr(header, "Flag_DoublePrecision", 1);
    write_attr(header, "HubbleParam", 1.0);
    write_attr(header, "Omega0", 0.0);      write_attr(header, "OmegaLambda", 0.0);
    H5Gclose(header);

    for (int t = 0; t < 6; ++t) {
        if (count_per_type[t] == 0) continue;
        const size_t at = type_start[t], n_type = count_per_type[t];
        char group_name[16];
        snprintf(group_name, sizeof group_name, "PartType%d", t);
        hid_t group = H5Gcreate2(file, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        auto write_vector_field = [&](const char* name, const std::vector<double>& comp_x,
                                      const std::vector<double>& comp_y,
                                      const std::vector<double>& comp_z) {
            hsize_t dims[2] = {n_type, 3};
            hid_t space = H5Screate_simple(2, dims, nullptr);
            hid_t dataset = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            std::vector<double> interleaved(3*n_type);
            for (size_t i = 0; i < n_type; ++i) {
                interleaved[3*i] = comp_x[at + i];
                interleaved[3*i+1] = comp_y[at + i];
                interleaved[3*i+2] = comp_z[at + i];
            }
            H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     interleaved.data());
            H5Dclose(dataset); H5Sclose(space);
        };
        auto write_scalar_field = [&](const char* name, const std::vector<double>& values) {
            hsize_t dims = n_type;
            hid_t space = H5Screate_simple(1, &dims, nullptr);
            hid_t dataset = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     values.data() + at);
            H5Dclose(dataset); H5Sclose(space);
        };
        write_vector_field("Coordinates", sim.P.x, sim.P.y, sim.P.z);
        // Raw stored velocity, exactly as the reference writes it (file_io/io.cc:230 -- "note
        // this is -not- the exact velocity in-code b/c we're alternating drifts and kicks!").
        // The half-kick offset this carries is the reference's own snapshot convention.
        write_vector_field("Velocities", sim.vx, sim.vy, sim.vz);
        write_scalar_field("Masses", sim.P.m);
        if (t == 0) {
            // PREDICTED internal energy, as the reference outputs (io.cc:276 writes
            // InternalEnergyPred): the stored u is half-kicked like the velocity, and completing
            // the owed half with the particle's own rate synchronises it. Exact at a particle's
            // own sync point; mid-step it is the same first-order prediction the reference makes.
            if (sim.du_dt.size() == n_part && sim.pending_half_kick.size() == n_part) {
                std::vector<double> u_pred(sim.u);
                for (size_t q = 0; q < n_part && q < sim.n_gas; ++q)
                    u_pred[q] = std::max(u_pred[q] + sim.du_dt[q] * sim.pending_half_kick[q],
                                         1e-30);
                write_scalar_field("InternalEnergy", u_pred);
            } else {
                write_scalar_field("InternalEnergy", sim.u);
            }
            write_scalar_field("Density", sim.rho);
            write_scalar_field("SmoothingLength", sim.h);
        }
        if (t == 5) {
            // Dataset names follow the reference (file_io/io.cc:4019,4036,4040,3860). Sink_Mass is
            // the STELLAR mass -- the dynamical mass less whatever has yet to drain out of the
            // unresolved disk -- so it compares directly against the reference's field of the same
            // name. There is still no protostellar evolution, so those fields remain absent.
            if (sim.sink_reservoir.size() == n_part) {
                std::vector<double> m_star(n_part);
                for (size_t q = 0; q < n_part; ++q)
                    m_star[q] = sim.P.m[q] - sim.sink_reservoir[q];
                write_scalar_field("Sink_Mass", m_star);
                write_scalar_field("Sink_Mass_Reservoir", sim.sink_reservoir);
            } else {
                write_scalar_field("Sink_Mass", sim.P.m);
            }
            if (sim.sink_radius.size() == n_part)
                write_scalar_field("Sink_Radius", sim.sink_radius);
            if (sim.sink_tform.size() == n_part)
                write_scalar_field("StellarFormationTime", sim.sink_tform);
            if (sim.sink_m0.size() == n_part)
                write_scalar_field("Sink_InitialMass", sim.sink_m0);
        }
        if (sim.output_potential && sim.phi.size() == n_part)
            write_scalar_field("Potential", sim.phi);
        {
            hsize_t dims = n_type;
            hid_t space = H5Screate_simple(1, &dims, nullptr);
            hid_t dataset = H5Dcreate2(group, "ParticleIDs", H5T_NATIVE_LLONG, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     particle_ids.data() + at);
            H5Dclose(dataset); H5Sclose(space);
        }
        H5Gclose(group);
    }
    H5Fclose(file);
    printf("wrote %s (t=%.6g)\n", filename, time); fflush(stdout);
}


// Write a snapshot AT a requested time that need not be a step boundary. Positions are drifted
// back by `back_dt` from wherever the step landed; velocities are constant between kicks, so
// this is exact rather than an interpolation error. The simulation state is restored afterwards,
// so producing output never perturbs the integration -- which is the property that lets the
// timestep ladder be anchored to the run instead of to the output cadence.
static void write_snapshot_at(Sim& sim, const std::vector<long long>& particle_ids,
                              const std::string& outdir, int snapshot_num, double out_time,
                              double back_dt, double header_box) {
    if (back_dt <= 0) {
        write_snapshot(sim, particle_ids, outdir, snapshot_num, out_time, header_box);
        return;
    }
    const size_t n = sim.size();
    std::vector<double> keep_x(sim.P.x), keep_y(sim.P.y), keep_z(sim.P.z);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double x = sim.P.x[i] - sim.vx[i] * back_dt;
        double y = sim.P.y[i] - sim.vy[i] * back_dt;
        double z = sim.P.z[i] - sim.vz[i] * back_dt;
        if (sim.box > 0) {
            x = fold_into_box(x, sim.box); y = fold_into_box(y, sim.box);
            z = fold_into_box(z, sim.box);
        }
        sim.P.x[i] = x; sim.P.y[i] = y; sim.P.z[i] = z;
    }
    write_snapshot(sim, particle_ids, outdir, snapshot_num, out_time, header_box);
    sim.P.x.swap(keep_x); sim.P.y.swap(keep_y); sim.P.z.swap(keep_z);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nranks = 1; MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Idle ranks finalize immediately rather than parking on a barrier: an OpenMPI barrier
    // busy-waits, so N-1 ranks spinning there would burn N-1 cores for the whole run -- the exact
    // __sched_yield pathology this engine exists to remove.
    if (rank != 0) { MPI_Finalize(); return 0; }
    if (argc < 2) { fprintf(stderr, "usage: %s <paramsfile> [restartflag]\n", argv[0]); return 2; }

    // Claim the footprint the job was actually given. The harness launches real GIZMO as
    // `mpirun -np R` with OMP_NUM_THREADS=T, i.e. R*T-way parallelism; here rank 0 is the only
    // worker, and mpirun has pinned it to just its own T hwthreads. Left alone, a 16x2 launch
    // would run this engine 2-way. So widen the affinity mask back to every online CPU and take
    // R*T threads -- the same total the job asked for, just all in one process.
    {
        const long n_cpus_online = sysconf(_SC_NPROCESSORS_ONLN);
        cpu_set_t every_cpu;
        CPU_ZERO(&every_cpu);
        for (long cpu = 0; cpu < n_cpus_online && cpu < CPU_SETSIZE; ++cpu) CPU_SET(cpu, &every_cpu);
        if (sched_setaffinity(0, sizeof(every_cpu), &every_cpu) != 0)
            fprintf(stderr, "warning: could not widen CPU affinity; staying on mpirun's mask\n");
        const char* omp_threads_env = getenv("OMP_NUM_THREADS");
        const int threads_per_rank = omp_threads_env ? std::max(1, atoi(omp_threads_env)) : 1;
        const int n_threads = std::min<long>((long)nranks * threads_per_rank, n_cpus_online);
        omp_set_num_threads(n_threads);
        printf("shmem-GIZMO: rank 0 of %d doing all work; %d OpenMP threads (%ld cpus online)\n",
               nranks, n_threads, n_cpus_online);
    }

    auto params = parse_kv(argv[1]);
    auto need = [&](const char* key)->std::string {
        auto it = params.find(key);
        if (it == params.end()) { fprintf(stderr, "params missing %s\n", key); exit(1); }
        return it->second;
    };
    const std::string icfile = need("InitCondFile") + ".hdf5";
    const std::string outdir = params.count("OutputDir") ? params["OutputDir"] : "output";
    const double time_max = atof(need("TimeMax").c_str());
    const double dt_snapshot =
        params.count("TimeBetSnapshot") ? atof(params["TimeBetSnapshot"].c_str()) : time_max;
    const double dt_max =
        params.count("MaxSizeTimestep") ? atof(params["MaxSizeTimestep"].c_str()) : 1e30;
    const double des_ngb = params.count("DesNumNgb") ? atof(params["DesNumNgb"].c_str()) : 32.0;
    const double courant = params.count("CourantFac") ? atof(params["CourantFac"].c_str()) : 0.2;
    const double box     = params.count("BoxSize") ? atof(params["BoxSize"].c_str()) : 0.0;

    int n_dims = 3; double gamma = 5.0/3.0;
    bool gravity_on = false, adaptive_soft = false, output_potential = false;
    bool tidal_criterion = false, box_periodic = false, sink_formation = false;
    double eos_adiabat = 0.0; int baro_variant = -1; bool baro_soundspeed = false;
    int hermite_mask = 0; bool cooling_on = false; bool hybrid_opening = false;
    bool developer_mode = false; bool galsf = false; bool randomize_gravtree = false;
    double atu_frac = 0.0;
    parse_config(n_dims, gamma, gravity_on, adaptive_soft, output_potential, tidal_criterion,
                 box_periodic, eos_adiabat, baro_variant, baro_soundspeed, sink_formation,
                 hermite_mask, cooling_on, hybrid_opening, developer_mode, galsf,
                 randomize_gravtree, atu_frac);
    // Units first: G falls back to the PHYSICAL constant in code units when the params file
    // leaves GravityConstantInternal at 0 or absent, which is GIZMO's documented behaviour
    // ("calculated by code if =0") and what every physical-units test relies on. Defaults match
    // GIZMO's own (kpc / 1e10 Msun / km s^-1), so a params file that omits them is unchanged.
    const double unit_length_cgs = params.count("UnitLength_in_cm")
                                 ? atof(params["UnitLength_in_cm"].c_str()) : 3.085678e21;
    const double unit_mass_cgs   = params.count("UnitMass_in_g")
                                 ? atof(params["UnitMass_in_g"].c_str()) : 1.989e43;
    const double unit_vel_cgs    = params.count("UnitVelocity_in_cm_per_s")
                                 ? atof(params["UnitVelocity_in_cm_per_s"].c_str()) : 1.0e5;
    const double GRAVITY_CGS = 6.674e-8;
    double grav_const = params.count("GravityConstantInternal")
                      ? atof(params["GravityConstantInternal"].c_str()) : 0.0;
    if (!(grav_const > 0))
        grav_const = GRAVITY_CGS * unit_mass_cgs / (unit_length_cgs * unit_vel_cgs * unit_vel_cgs);
    // Softening params come in two GIZMO spellings; accept both. Values in the file are
    // Plummer-equivalent; the engine stores the KERNEL EXTENT = 2.8x (GIZMO's ForceSoftening).
    auto soft_param = [&](const char* name_a, const char* name_b) -> double {
        if (params.count(name_a)) return atof(params[name_a].c_str());
        if (params.count(name_b)) return atof(params[name_b].c_str());
        return 0.0;
    };
    const double soft_plummer[6] = {
        soft_param("Softening_Type0", "SofteningGas"),
        soft_param("Softening_Type1", "SofteningHalo"),
        soft_param("Softening_Type2", "SofteningDisk"),
        soft_param("Softening_Type3", "SofteningBulge"),
        soft_param("Softening_Type4", "SofteningStars"),
        soft_param("Softening_Type5", "SofteningBndry")};
    printf("shmem-GIZMO: %s  dim=%d gamma=%.6f box=%g%s TimeMax=%g DesNumNgb=%g CFL=%g\n",
           icfile.c_str(), n_dims, gamma, box, box_periodic ? " (periodic)" : "",
           time_max, des_ngb, courant);
    printf("shmem-GIZMO: gravity=%s%s G=%g soft=(%g,%g)\n",
           gravity_on ? "on" : "off", (gravity_on && adaptive_soft) ? " (adaptive)" : "",
           grav_const, soft_plummer[0], soft_plummer[1]);
    fflush(stdout);

    // load ICs
    hid_t ic_file = H5Fopen(icfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (ic_file < 0) { fprintf(stderr, "cannot open %s\n", icfile.c_str()); return 1; }
    Sim sim;
    sim.dim = n_dims; sim.gamma = gamma;
    // Wrapping is keyed on the CONFIG flag, not on BoxSize: plummer/shu1977 carry a nonzero
    // BoxSize through an open (non-periodic) run. The header still records the params value.
    sim.box = box_periodic ? box : 0.0;
    sim.des_ngb = des_ngb; sim.cfl = courant;
    sim.gravity_on = gravity_on; sim.G = grav_const;
    sim.soft_min = 2.8 * soft_plummer[0]; sim.adaptive_soft = adaptive_soft;
    for (int t = 0; t < 6; ++t) sim.soft_fixed[t] = 2.8 * soft_plummer[t];
    sim.output_potential = output_potential;
    sim.tidal_criterion = tidal_criterion;
    sim.hermite_mask = hermite_mask;
    if (hermite_mask)
        printf("shmem-GIZMO: hermite integration on for type mask %d\n", hermite_mask);

    // ---- EOS ----
    // The barotropic constants are tabulated against n_H in cm^-3 and return cgs pressure, so
    // both conversions are precomputed from the unit system read above.
    const double unit_density_cgs = unit_mass_cgs / (unit_length_cgs*unit_length_cgs*unit_length_cgs);
    const double PROTONMASS = 1.6726e-24, HYDROGEN_MASSFRAC = 0.76;
    sim.nh_per_code_density = unit_density_cgs * HYDROGEN_MASSFRAC / PROTONMASS;
    sim.code_press_per_cgs  = 1.0 / (unit_density_cgs * unit_vel_cgs * unit_vel_cgs);
    if (baro_variant >= 0) {
        sim.eos_law = Sim::EosLaw::BAROTROPIC;
        sim.baro_variant = baro_variant;
        sim.baro_soundspeed = baro_soundspeed;
    } else if (eos_adiabat > 0) {
        sim.eos_law = Sim::EosLaw::ENFORCE_ADIABAT;
        sim.eos_adiabat = eos_adiabat;
    }
    sim.mass_to_solar = unit_mass_cgs / 1.989e33;
    sim.vel_to_kms = unit_vel_cgs / 1.0e5;
    sim.length_to_au = unit_length_cgs / 1.495978707e13;
    sim.opacity_limit_physics = cooling_on || (baro_variant >= 0);
    sim.sink_formation = sink_formation;
    sim.hybrid_opening = hybrid_opening;
    sim.randomize_gravtree = randomize_gravtree;
    // Config-driven, as the reference is. MEASURED on evrard (27k particles, adiabatic collapse),
    // sweeping the fraction with everything else fixed -- the engine is bit-reproducible at fixed
    // thread count (two baselines agreed to 8e-15), so these are signal:
    //
    //     frac     steps   wall    grav    rho error vs base
    //     base     1010    96.1s   35.1s   --
    //     0.0625   1010    96.1s   36.1s   1.7e-02      <- reference default: skips nothing here
    //     0.5      1006    71.6s   12.0s   6.1e-02
    //     2         986    64.4s    5.7s   1.0e-01
    //     8         904    60.3s    3.8s   3.6e-01
    //     32        800    60.0s    2.0s   8.6e-01
    //
    // So the scheme works, and the FRACTION is the knob. 0.0625 is tuned for runs whose steps are
    // set by multiphysics (radiation, feedback), where dt << t_tidal already -- gravity-limited
    // problems like this one need a larger value before anything is skipped, and pay a jerk per
    // walk in the meantime. The accuracy cost is jerk extrapolation error and rises steeply past 2.
    sim.atu_frac = atu_frac;
    if (atu_frac > 0)
        printf("shmem-GIZMO: adaptive tree-force update ON (gas keeps a jerk-advanced force for "
               "%.4g of its tidal time)\n", atu_frac);
    // Sink-sink direct summation, on wherever the sink bundle is (the reference gates it on
    // SINGLE_STAR_TIMESTEPPING or SINGLE_STAR_FIND_BINARIES, both of which ride in that bundle).
    // 1000 AU converted to code units; sinks are the only type it applies to, so a run without
    // them never pays for it.
    if (sink_formation && sim.length_to_au > 0)
        sim.sink_direct_radius = 1000.0 / sim.length_to_au;
    // Contact-wave signal velocity, ON wherever the reference applies it: MFM + GALSF
    // (hydro_core_meshless.h:252-254). This engine is MFM-only, so the config's GALSF decides.
    //
    // It costs about 10% more steps: 3753 against 3402 to reach t=0.002 on shu1977, two replicates
    // each and exactly reproducible. 2*S_M runs somewhat above the Monaghan cs_i+cs_j, so the
    // Courant step tightens -- but only modestly, not the factor of ~2 the raw dt values at a
    // couple of sampled points suggest. Count steps from the run's own "done: ... in N steps"
    // total; the progress lines are a 10-second wall-clock heartbeat, so comparing the last one
    // between runs of different length compares different fractions of each run.
    // SHMEM_NO_CONTACT_VSIG forces the Monaghan estimate for A/B work.
    sim.contact_wave_vsig = galsf && (getenv("SHMEM_NO_CONTACT_VSIG") == nullptr);
    if (sink_formation) {
        // CritPhysDensity is in n_H cm^-3; PhysDensThresh is the same in code density units.
        const double crit_nh = params.count("CritPhysDensity")
                             ? atof(params["CritPhysDensity"].c_str()) : 0.0;
        sim.crit_phys_density = (sim.nh_per_code_density > 0)
                              ? crit_nh / sim.nh_per_code_density : 0.0;
        // MaxSfrTimescale IS NOT A TIMESCALE. GIZMO registers two params at the same address
        // (core/begrun.cc:1758 and 1771) -- the legacy `MaxSfrTimescale` and the one that is
        // actually meant, `SfEffPerFreeFall`, a DIMENSIONLESS EFFICIENCY -- and then converts it
        // at begrun.cc:572 into the free-fall time at the CRITICAL density over that efficiency:
        //     All.MaxSfrTimescale = (1/eff) * sqrt(3 pi / (32 G rho_crit))
        // Combined with tsfr = sqrt(rho_crit/rho) * All.MaxSfrTimescale (sfr_eff.cc:203) this is
        // just tsfr = t_ff(LOCAL rho) / eff.
        //
        // Reading the legacy value as a timescale is what stopped this port ever forming a sink:
        // shu1977.params carries SfEffPerFreeFall 1.0 AND MaxSfrTimescale 4000, so tsfr came out
        // ~2757 instead of 3.8e-5 -- 7.3e7 too large. The time-averaged virial criterion (&2048)
        // grows AlphaVirial_SF_TimeSmoothed from 0 by ~8*dt/tsfr per step, so that error meant
        // ~4.5e10 steps to converge rather than ~600, and no cell could ever pass.
        const double sf_eff = params.count("SfEffPerFreeFall")
                            ? atof(params["SfEffPerFreeFall"].c_str()) : 1.0;
        sim.max_sfr_timescale = (sf_eff > 0 && sim.crit_phys_density > 0 && sim.G > 0)
            ? std::sqrt(3.0*M_PI / (32.0 * sim.G * sim.crit_phys_density)) / sf_eff : 0.0;
        printf("shmem-GIZMO: sink formation ON  (PhysDensThresh = %.6g code = %.6g cm^-3, "
               "SfEffPerFreeFall = %g, t_ff(rho_crit)/eff = %g)\n",
               sim.crit_phys_density, crit_nh, sf_eff, sim.max_sfr_timescale);
    }
    if (!sim.eos_is_ideal())
        printf("shmem-GIZMO: EOS %s%s  (n_H per code rho = %.4g cm^-3, P_code per P_cgs = %.4g)\n",
               sim.eos_law == Sim::EosLaw::ENFORCE_ADIABAT ? "enforced adiabat P = A rho^gamma"
                                                           : "GMC barotropic",
               sim.eos_law == Sim::EosLaw::BAROTROPIC
                   ? (sim.baro_soundspeed ? " variant/soundspeed-from-barotrope" : " variant")
                   : "",
               sim.nh_per_code_density, sim.code_press_per_cgs);
    if (params.count("ErrTolIntAccuracy")) sim.eta_grav = atof(params["ErrTolIntAccuracy"].c_str());
    if (params.count("ErrTolTheta"))    sim.theta = atof(params["ErrTolTheta"].c_str());
    if (params.count("MaxNumNgbDeviation"))
        sim.ngb_tol = atof(params["MaxNumNgbDeviation"].c_str());
    if (params.count("ErrTolForceAcc")) sim.err_tol_force_acc = atof(params["ErrTolForceAcc"].c_str());
    // WITHOUT DEVELOPER_MODE the reference IGNORES these accuracy parameters and substitutes its
    // own (core/begrun.cc:2637-2677, which warns "Tag ... was specified, but it is being ignored").
    // Reading them from the file instead is not a small difference: shu1977 asks for CourantFac
    // 0.2 and ErrTolTheta 0.21 while the reference runs 0.4 and 0.5, so every Courant step here
    // was half the reference's and every tree walk opened more nodes than its does.
    if (!developer_mode) {
        sim.cfl = 0.4; sim.eta_grav = 0.02; sim.theta = 0.7; sim.err_tol_force_acc = 0.0025;
        // ...then the accurate-few-body tightenings on top (begrun.cc:2744-2748), which the
        // STARFORGE bundle turns on via GRAVITY_ACCURATE_FEWBODY_INTEGRATION.
        if (hybrid_opening) {
            if (sim.eta_grav > 0.01) sim.eta_grav = 0.01;
            if (sim.theta    > 0.5)  sim.theta    = 0.5;
            if (sim.ngb_tol  > 0.05) sim.ngb_tol  = 0.05;
        }
        printf("shmem-GIZMO: no DEVELOPER_MODE -- using the reference's built-in accuracy "
               "parameters (CourantFac=%g ErrTolIntAccuracy=%g ErrTolTheta=%g ErrTolForceAcc=%g)\n",
               sim.cfl, sim.eta_grav, sim.theta, sim.err_tol_force_acc);
    }
    // SHMEM_GLOBAL_TIMESTEP=1 forces the old all-active scheme, for A/B against this one.
    sim.individual_timesteps = (getenv("SHMEM_GLOBAL_TIMESTEP") == nullptr);
    // SHMEM_DENSE_DRIFT=1 restores the full O(N) drift sweep, for A/B against lazy drift.
    sim.sparse_drift = (getenv("SHMEM_DENSE_DRIFT") == nullptr);
    if (const char* wf = getenv("SHMEM_WAKEUP_FAC")) sim.wakeup_fac = std::max(1.0, atof(wf));
    // SHMEM_TREE_PAD_FRAC: how far the tree may go stale before a rebuild. The gravity walk
    // reads node centres-of-mass as they were AT BUILD TIME, so this directly controls a force
    // error that the neighbour-search padding does not cover. 0 rebuilds every sync, for A/B.
    if (const char* tp = getenv("SHMEM_TREE_PAD_FRAC")) sim.tree_rebuild_pad_frac = atof(tp);
    // Anchor the integer timeline on the WHOLE RUN, exactly as the reference does
    // (core/init.cc:114, Timebase_interval = (TimeMax - TimeBegin)/TIMEBASE). Anchoring it on
    // the SNAPSHOT interval instead -- which this used to do -- makes the reachable timesteps
    // depend on the output cadence: two runs of the same problem written at different
    // TimeBetSnapshot integrate on different ladders, and neither matches GIZMO's unless the
    // snapshot interval happens to be a power-of-two fraction of the run. With TimeMax=0.02 and
    // MaxSizeTimestep=0.005 this gives dt_base = 0.005 = 0.02/4, so the ladder is 0.02/2^k --
    // identical to the reference's reachable set.
    set_time_base(sim, time_max - 0.0, dt_max);
    printf("shmem-GIZMO: timesteps=%s dt_base=%g\n",
           sim.individual_timesteps ? "individual" : "global", sim.dt_base);

    // Every particle type present, in ascending type order -- which is what gives the engine its
    // gas-first index layout (Sim::n_gas). Masses fall back to the header MassTable when a group
    // carries no Masses dataset; InternalEnergy exists for gas only.
    // MassTable is OPTIONAL. All-zero is the common case anyway ("masses are in the dataset"), and
    // a minimal writer may omit it entirely -- MakeCloud does, so every STARFORGE cloud IC lacks
    // it, as does the bate M50 production run's. Opening it unconditionally aborts the read on
    // those files. Absent means zero, which is the same instruction.
    double ic_mass_table[6] = {0,0,0,0,0,0};
    {
        hid_t header = H5Gopen2(ic_file, "Header", H5P_DEFAULT);
        if (H5Aexists(header, "MassTable") > 0) {
            hid_t attr = H5Aopen(header, "MassTable", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, ic_mass_table);
            H5Aclose(attr);
        }
        H5Gclose(header);
    }
    std::vector<long long>& particle_ids = sim.id;
    std::vector<uint8_t> loaded_types;
    auto append = [](std::vector<double>& dst, const std::vector<double>& src) {
        dst.insert(dst.end(), src.begin(), src.end());
    };
    for (int t = 0; t < 6; ++t) {
        char group_name[16];
        snprintf(group_name, sizeof group_name, "PartType%d", t);
        if (H5Lexists(ic_file, group_name, H5P_DEFAULT) <= 0) continue;
        hid_t group = H5Gopen2(ic_file, group_name, H5P_DEFAULT);
        const std::vector<double> xs = h5_read(group, "Coordinates", 0);
        const size_t n_type = xs.size();
        append(sim.P.x, xs);
        append(sim.P.y, h5_read(group, "Coordinates", 1));
        append(sim.P.z, h5_read(group, "Coordinates", 2));
        if (H5Lexists(group, "Masses", H5P_DEFAULT) > 0) {
            append(sim.P.m, h5_read(group, "Masses", -1));
        } else {
            sim.P.m.insert(sim.P.m.end(), n_type, ic_mass_table[t]);
        }
        append(sim.vx, h5_read(group, "Velocities", 0));
        append(sim.vy, h5_read(group, "Velocities", 1));
        append(sim.vz, h5_read(group, "Velocities", 2));
        if (t == 0) append(sim.u, h5_read(group, "InternalEnergy", -1));
        else        sim.u.insert(sim.u.end(), n_type, 0.0);
        // Per-particle h guesses for the initial solve. Without them density() falls back to ONE
        // global-mean spacing, and on a restart of an evolved state (6+ decades of density) that
        // guess is ~7x too large for the typical particle: the init h-solve then burns ~7 full
        // searches per target at ~350x the right neighbour count -- measured as an HOUR of
        // startup at 3.5e6 cells. Prefer the snapshot's own SmoothingLength; fall back to the
        // density-derived scale (also right everywhere); only a bare IC gets the global default.
        if (t == 0) {
            if (H5Lexists(group, "SmoothingLength", H5P_DEFAULT) > 0) {
                append(sim.h, h5_read(group, "SmoothingLength", -1));
            } else if (H5Lexists(group, "Density", H5P_DEFAULT) > 0) {
                const std::vector<double> rho0 = h5_read(group, "Density", -1);
                const size_t base = sim.h.size();
                sim.h.resize(base + n_type, 0.0);
                for (size_t q = 0; q < n_type; ++q) {
                    const double m_q = sim.P.m[base + q];
                    if (rho0[q] > 0 && m_q > 0)
                        sim.h[base + q] = std::cbrt(3.0 * 32.0 * m_q / (4.0 * M_PI * rho0[q]));
                }
            }
        }
        const std::vector<double> ids_as_double = h5_read(group, "ParticleIDs", -1);
        for (double id : ids_as_double) particle_ids.push_back((long long)id);
        loaded_types.insert(loaded_types.end(), n_type, (uint8_t)t);
        if (t == 0) sim.n_gas = n_type;
        H5Gclose(group);
        printf("shmem-GIZMO: loaded %zu particles of type %d\n", n_type, t);
    }
    H5Fclose(ic_file);
    if (sim.size() == 0) { fprintf(stderr, "no particles in %s\n", icfile.c_str()); return 1; }
    if (sim.n_gas == (size_t)-1) sim.n_gas = 0;          // no PartType0 group at all
    if (sim.n_gas < sim.size()) sim.P.type = loaded_types;  // empty type list means "all gas"
    sim.P.soft.assign(sim.size(), 0.0);

    (void)system(("mkdir -p " + outdir).c_str());

    // snapshot 0 needs Density/h populated. Use the engine's own volume solve, NOT the SPH-style
    // sum_j m_j W that the neighbour routine returns -- see compute_initial_state.
    compute_initial_state(sim);
    if (sim.output_potential) compute_potential(sim);
    sync_all_positions(sim);
    write_snapshot(sim, particle_ids, outdir, 0, 0.0, box);

    // Time is tracked in the engine's INTEGER TICKS, not accumulated in floating point. Summing
    // dt every step leaves a residual at each snapshot boundary, and asking mfm_step to close a
    // residual smaller than one tick used to desynchronise the whole timestep hierarchy (see
    // Sim::ticks_floor). Integer targets make every boundary exact and every step a whole
    // number of ticks, so the question never arises.
    double time = 0; int snapshot_num = 1; int n_steps = 0;
    const long long end_ticks = sim.individual_timesteps ? sim.ticks_of_time(time_max) : 0;
    const long long snap_ticks = sim.individual_timesteps ? sim.ticks_of_time(dt_snapshot) : 0;
    long long next_snapshot_ticks = snap_ticks;
    double next_snapshot_time = dt_snapshot;     // global-timestep mode keeps the float path
    const auto wall_start = std::chrono::steady_clock::now();
    // Heartbeat on WALL CLOCK, not step count: a step-count heartbeat stays silent until it first
    // fires, so the slower the run the longer it reports nothing -- backwards from what is wanted.
    double next_report = 10.0;
    while (sim.individual_timesteps ? (sim.clock_ticks < end_ticks) : (time < time_max - 1e-12)) {
        double dt_allowed;
        if (sim.individual_timesteps) {
            // Only the END of the run bounds the step now. Snapshots are produced by drifting
            // to the output time (below), not by truncating the step to hit it -- truncation is
            // what forced the ladder to be tied to the output cadence.
            dt_allowed = std::min(dt_max, sim.time_of_ticks(end_ticks - sim.clock_ticks));
        } else {
            dt_allowed = std::min(dt_max, std::min(next_snapshot_time, time_max) - time);
        }
        const double dt_taken = mfm_step(sim, dt_allowed);
        time = sim.individual_timesteps ? sim.time_now() : time + dt_taken;
        ++n_steps;
        const double wall_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        if (wall_elapsed >= next_report) {
            next_report = wall_elapsed + 10.0;
            const double frac = time / time_max;
            printf("  step %d  t=%.5g/%g (%.1f%%)  dt=%.3g  %.0f ms/step  eta %.1f min\n",
                   n_steps, time, time_max, 100.0*frac, dt_taken,
                   1e3 * wall_elapsed / n_steps,
                   frac > 0 ? wall_elapsed * (1.0/frac - 1.0) / 60.0 : -1.0);
            fflush(stdout);
            // On the same heartbeat rather than every sync: GIZMO dumps this per sync point, which
            // here would be thousands of tables. This keeps it readable while still showing how the
            // hierarchy evolves as the run proceeds.
            if (sim.individual_timesteps) print_timebins(sim, dt_taken, time);
        }
        energy_log_step(sim, time);   // SHMEM_ENERGY_LOG; no-op unless the env var is set
        // OUTPUT. A step may now overshoot -- or leap clean over -- one or more output times,
        // so this is a loop, and each snapshot is written at its OWN requested time by drifting
        // positions there rather than at wherever the step happened to land. Velocities are
        // constant between kicks, so the interpolation is exact, not an approximation; this is
        // GIZMO's strategy (drift to All.Ti_nextoutput, write, carry on) with the state left
        // untouched afterwards. It also means an arbitrary list of output times costs nothing
        // extra, which is what OutputListOn will need.
        while (sim.individual_timesteps && sim.clock_ticks >= next_snapshot_ticks
               && next_snapshot_ticks < end_ticks) {
            if (sim.output_potential) compute_potential(sim);
            sync_all_positions(sim);
            write_snapshot_at(sim, particle_ids, outdir, snapshot_num++,
                              sim.time_of_ticks(next_snapshot_ticks),
                              sim.time_of_ticks(sim.clock_ticks - next_snapshot_ticks), box);
            next_snapshot_ticks += snap_ticks;
        }
        if (!sim.individual_timesteps &&
            time >= std::min(next_snapshot_time, time_max) - 1e-12 && next_snapshot_time < time_max) {
            if (sim.output_potential) compute_potential(sim);
            sync_all_positions(sim);
            write_snapshot(sim, particle_ids, outdir, snapshot_num++, time, box);
            next_snapshot_time += dt_snapshot;
        }
    }
    if (sim.output_potential) compute_potential(sim);
    sync_all_positions(sim);
    write_snapshot(sim, particle_ids, outdir, snapshot_num, time, box);
    printf("done: t=%.6g in %d steps\n", time, n_steps);
    hermite_report();
    MPI_Finalize();
    return 0;
}
