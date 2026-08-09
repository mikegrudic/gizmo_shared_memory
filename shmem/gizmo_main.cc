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
        if (sscanf(line, "%127s %255s", key, value) == 2) settings[key] = value;
    }
    fclose(file);
    return settings;
}

// EOS_GAMMA=(5.0/3.0) / BOX_SPATIAL_DIMENSION=2 out of whichever Config.sh the harness staged
static void parse_config(int& n_dims, double& gamma, bool& gravity_on, bool& adaptive_soft,
                         bool& output_potential, bool& tidal_criterion) {
    // Gravity is ON in GIZMO unless SELFGRAVITY_OFF is set, so default to on and let the config
    // switch it off -- the opposite default would silently drop gravity from any test whose
    // Config.sh simply does not mention it.
    gravity_on = true;
    adaptive_soft = false;
    output_potential = false;
    tidal_criterion = false;
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
            // ...except the ones this engine satisfies unconditionally, which would otherwise
            // make the warning pure noise: MFM is what the engine IS, output is always double,
            // and DEVELOPER_MODE only exposes extra params (already read by name).
            for (const char* benign : {"HYDRO_MESHLESS_FINITE_MASS", "OUTPUT_IN_DOUBLEPRECISION",
                                       "DEVELOPER_MODE"})
                flag(benign);
            // Flags this engine has no implementation for. Reported rather than ignored in
            // silence: a suite variant exists precisely to exercise the feature its flag names,
            // so running it as if the flag were absent makes the variant a duplicate of baseline
            // that PASSES -- which is how plummer/tidal and evrard/tidal_adaptive went green
            // without ever enabling the criterion they are named after.
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
                           const std::string& outdir, int snapshot_num, double time) {
    char filename[512];
    snprintf(filename, sizeof filename, "%s/snapshot_%03d.hdf5", outdir.c_str(), snapshot_num);
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    const size_t n_part = sim.size();

    hid_t header = H5Gcreate2(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    unsigned int count_per_type[6] = {(unsigned)n_part,0,0,0,0,0}, zeros[6] = {0,0,0,0,0,0};
    int count_per_type_int[6] = {(int)n_part,0,0,0,0,0};
    double mass_table[6] = {0,0,0,0,0,0};       // 0 => per-particle masses live in the dataset
    write_attr_per_type(header, "NumPart_ThisFile", H5T_NATIVE_INT, count_per_type_int);
    write_attr_per_type(header, "NumPart_Total", H5T_NATIVE_UINT, count_per_type);
    write_attr_per_type(header, "NumPart_Total_HighWord", H5T_NATIVE_UINT, zeros);
    write_attr_per_type(header, "MassTable", H5T_NATIVE_DOUBLE, mass_table);
    write_attr(header, "Time", time);
    write_attr(header, "Redshift", 0.0);
    write_attr(header, "BoxSize", sim.box);
    write_attr(header, "NumFilesPerSnapshot", 1);
    write_attr(header, "Flag_Sfr", 0);      write_attr(header, "Flag_Cooling", 0);
    write_attr(header, "Flag_Feedback", 0); write_attr(header, "Flag_StellarAge", 0);
    write_attr(header, "Flag_Metals", 0);   write_attr(header, "Flag_DoublePrecision", 1);
    write_attr(header, "HubbleParam", 1.0);
    write_attr(header, "Omega0", 0.0);      write_attr(header, "OmegaLambda", 0.0);
    H5Gclose(header);

    hid_t gas_group = H5Gcreate2(file, "PartType0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    auto write_vector_field = [&](const char* name, const std::vector<double>& comp_x,
                                  const std::vector<double>& comp_y,
                                  const std::vector<double>& comp_z) {
        hsize_t dims[2] = {n_part, 3};
        hid_t space = H5Screate_simple(2, dims, nullptr);
        hid_t dataset = H5Dcreate2(gas_group, name, H5T_NATIVE_DOUBLE, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        std::vector<double> interleaved(3*n_part);
        for (size_t i = 0; i < n_part; ++i) {
            interleaved[3*i] = comp_x[i];
            interleaved[3*i+1] = comp_y[i];
            interleaved[3*i+2] = comp_z[i];
        }
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, interleaved.data());
        H5Dclose(dataset); H5Sclose(space);
    };
    auto write_scalar_field = [&](const char* name, const std::vector<double>& values) {
        hsize_t dims = n_part;
        hid_t space = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(gas_group, name, H5T_NATIVE_DOUBLE, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Dclose(dataset); H5Sclose(space);
    };
    write_vector_field("Coordinates", sim.P.x, sim.P.y, sim.P.z);
    write_vector_field("Velocities", sim.vx, sim.vy, sim.vz);
    write_scalar_field("Masses", sim.P.m);
    write_scalar_field("InternalEnergy", sim.u);
    write_scalar_field("Density", sim.rho);
    write_scalar_field("SmoothingLength", sim.h);
    if (sim.output_potential && sim.phi.size() == n_part) write_scalar_field("Potential", sim.phi);
    {
        hsize_t dims = n_part;
        hid_t space = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(gas_group, "ParticleIDs", H5T_NATIVE_LLONG, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, particle_ids.data());
        H5Dclose(dataset); H5Sclose(space);
    }
    H5Gclose(gas_group); H5Fclose(file);
    printf("wrote %s (t=%.6g)\n", filename, time); fflush(stdout);
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
    bool tidal_criterion = false;
    parse_config(n_dims, gamma, gravity_on, adaptive_soft, output_potential, tidal_criterion);
    const double grav_const = params.count("GravityConstantInternal")
                            ? atof(params["GravityConstantInternal"].c_str()) : 1.0;
    const double soft_gas = params.count("SofteningGas")
                          ? atof(params["SofteningGas"].c_str()) : 0.0;
    printf("shmem-GIZMO: %s  dim=%d gamma=%.6f box=%g TimeMax=%g DesNumNgb=%g CFL=%g\n",
           icfile.c_str(), n_dims, gamma, box, time_max, des_ngb, courant);
    printf("shmem-GIZMO: gravity=%s%s G=%g SofteningGas=%g\n",
           gravity_on ? "on" : "off", (gravity_on && adaptive_soft) ? " (adaptive)" : "",
           grav_const, soft_gas);
    fflush(stdout);

    // load ICs
    hid_t ic_file = H5Fopen(icfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (ic_file < 0) { fprintf(stderr, "cannot open %s\n", icfile.c_str()); return 1; }
    hid_t gas_group = H5Gopen2(ic_file, "PartType0", H5P_DEFAULT);
    Sim sim;
    sim.dim = n_dims; sim.gamma = gamma; sim.box = box;
    sim.des_ngb = des_ngb; sim.cfl = courant;
    sim.gravity_on = gravity_on; sim.G = grav_const;
    sim.soft_min = soft_gas; sim.adaptive_soft = adaptive_soft;
    sim.output_potential = output_potential;
    sim.tidal_criterion = tidal_criterion;
    if (params.count("ErrTolIntAccuracy")) sim.eta_grav = atof(params["ErrTolIntAccuracy"].c_str());
    if (params.count("ErrTolTheta"))    sim.theta = atof(params["ErrTolTheta"].c_str());
    if (params.count("ErrTolForceAcc")) sim.err_tol_force_acc = atof(params["ErrTolForceAcc"].c_str());
    // SHMEM_GLOBAL_TIMESTEP=1 forces the old all-active scheme, for A/B against this one.
    sim.individual_timesteps = (getenv("SHMEM_GLOBAL_TIMESTEP") == nullptr);
    if (const char* bl = getenv("SHMEM_BIN_LIMIT")) sim.bin_limit = std::max(1, atoi(bl));
    set_time_base(sim, dt_snapshot, dt_max);
    printf("shmem-GIZMO: timesteps=%s dt_base=%g\n",
           sim.individual_timesteps ? "individual" : "global", sim.dt_base);
    sim.P.x = h5_read(gas_group, "Coordinates", 0);
    sim.P.y = h5_read(gas_group, "Coordinates", 1);
    sim.P.z = h5_read(gas_group, "Coordinates", 2);
    sim.P.m = h5_read(gas_group, "Masses", -1);
    sim.vx  = h5_read(gas_group, "Velocities", 0);
    sim.vy  = h5_read(gas_group, "Velocities", 1);
    sim.vz  = h5_read(gas_group, "Velocities", 2);
    sim.u   = h5_read(gas_group, "InternalEnergy", -1);
    const std::vector<double> ids_as_double = h5_read(gas_group, "ParticleIDs", -1);
    H5Gclose(gas_group); H5Fclose(ic_file);
    sim.P.soft.assign(sim.size(), 0.0);
    std::vector<long long> particle_ids(sim.size());
    for (size_t i = 0; i < sim.size(); ++i) particle_ids[i] = (long long)ids_as_double[i];

    (void)system(("mkdir -p " + outdir).c_str());

    // snapshot 0 needs Density/h populated. Use the engine's own volume solve, NOT the SPH-style
    // sum_j m_j W that the neighbour routine returns -- see compute_initial_state.
    compute_initial_state(sim);
    if (sim.output_potential) compute_potential(sim);
    write_snapshot(sim, particle_ids, outdir, 0, 0.0);

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
            const long long target_ticks = std::min(next_snapshot_ticks, end_ticks);
            dt_allowed = std::min(dt_max, sim.time_of_ticks(target_ticks - sim.clock_ticks));
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
        const bool at_snapshot = sim.individual_timesteps
                               ? (sim.clock_ticks >= next_snapshot_ticks && sim.clock_ticks < end_ticks)
                               : (time >= std::min(next_snapshot_time, time_max) - 1e-12 &&
                                  next_snapshot_time < time_max);
        if (at_snapshot) {
            if (sim.output_potential) compute_potential(sim);
            write_snapshot(sim, particle_ids, outdir, snapshot_num++, time);
            next_snapshot_ticks += snap_ticks;
            next_snapshot_time += dt_snapshot;
        }
    }
    if (sim.output_potential) compute_potential(sim);
    write_snapshot(sim, particle_ids, outdir, snapshot_num, time);
    printf("done: t=%.6g in %d steps\n", time, n_steps);
    MPI_Finalize();
    return 0;
}
