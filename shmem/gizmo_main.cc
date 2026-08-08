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

#include "mfm.h"
#include <hdf5.h>
#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace shmem;

static std::map<std::string, std::string> parse_kv(const char* path) {
    std::map<std::string, std::string> kv;
    FILE* f = fopen(path, "r");
    if (!f) return kv;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char k[128], v[256];
        if (line[0] == '%' || line[0] == '#') continue;
        if (sscanf(line, "%127s %255s", k, v) == 2) kv[k] = v;
    }
    fclose(f);
    return kv;
}

// EOS_GAMMA=(5.0/3.0) / BOX_SPATIAL_DIMENSION=2 out of whichever Config.sh the harness staged
static void parse_config(int& dim, double& gamma) {
    for (const char* p : {"Config.sh", "../../Config.sh"}) {
        FILE* f = fopen(p, "r");
        if (!f) continue;
        char line[512];
        while (fgets(line, sizeof line, f)) {
            if (line[0] == '#') continue;
            int d;
            if (sscanf(line, "BOX_SPATIAL_DIMENSION=%d", &d) == 1) dim = d;
            double a, b;
            if (sscanf(line, "EOS_GAMMA=(%lf/%lf)", &a, &b) == 2) gamma = a / b;
            else if (sscanf(line, "EOS_GAMMA=(%lf)", &a) == 1) gamma = a;
            else if (sscanf(line, "EOS_GAMMA=%lf", &a) == 1) gamma = a;
        }
        fclose(f);
        break;                                   // first Config.sh found wins (cwd over root)
    }
}

static std::vector<double> h5_read(hid_t g, const char* name, int col) {
    hid_t d = H5Dopen2(g, name, H5P_DEFAULT);
    hid_t sp = H5Dget_space(d);
    hsize_t dims[2] = {0, 0};
    int nd = H5Sget_simple_extent_ndims(sp);
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    size_t n = dims[0];
    std::vector<double> out(n);
    if (nd == 1) H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    else {
        std::vector<double> buf(n * dims[1]);
        H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        for (size_t i = 0; i < n; ++i) out[i] = buf[i*dims[1] + (col < 0 ? 0 : col)];
    }
    H5Sclose(sp); H5Dclose(d);
    return out;
}

static void attr_d(hid_t h, const char* n, double v) {
    hid_t s = H5Screate(H5S_SCALAR), a = H5Acreate2(h, n, H5T_NATIVE_DOUBLE, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &v); H5Aclose(a); H5Sclose(s);
}
static void attr_i(hid_t h, const char* n, int v) {
    hid_t s = H5Screate(H5S_SCALAR), a = H5Acreate2(h, n, H5T_NATIVE_INT, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT, &v); H5Aclose(a); H5Sclose(s);
}
template <typename T>
static void attr_v6(hid_t h, const char* n, hid_t type, const T* v) {
    hsize_t six = 6;
    hid_t s = H5Screate_simple(1, &six, nullptr);
    hid_t a = H5Acreate2(h, n, type, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, type, v); H5Aclose(a); H5Sclose(s);
}

static void write_snapshot(const Sim& S, const std::vector<long long>& ids,
                           const std::string& outdir, int num, double time) {
    char fn[512];
    snprintf(fn, sizeof fn, "%s/snapshot_%03d.hdf5", outdir.c_str(), num);
    hid_t f = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    const size_t n = S.size();

    hid_t hdr = H5Gcreate2(f, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    unsigned int np[6] = {(unsigned)n,0,0,0,0,0}, zero6u[6] = {0,0,0,0,0,0};
    int np_i[6] = {(int)n,0,0,0,0,0};
    double mt[6] = {0,0,0,0,0,0};
    attr_v6(hdr, "NumPart_ThisFile", H5T_NATIVE_INT, np_i);
    attr_v6(hdr, "NumPart_Total", H5T_NATIVE_UINT, np);
    attr_v6(hdr, "NumPart_Total_HighWord", H5T_NATIVE_UINT, zero6u);
    attr_v6(hdr, "MassTable", H5T_NATIVE_DOUBLE, mt);
    attr_d(hdr, "Time", time);
    attr_d(hdr, "Redshift", 0.0);
    attr_d(hdr, "BoxSize", S.box);
    attr_i(hdr, "NumFilesPerSnapshot", 1);
    attr_i(hdr, "Flag_Sfr", 0); attr_i(hdr, "Flag_Cooling", 0);
    attr_i(hdr, "Flag_Feedback", 0); attr_i(hdr, "Flag_StellarAge", 0);
    attr_i(hdr, "Flag_Metals", 0); attr_i(hdr, "Flag_DoublePrecision", 1);
    attr_d(hdr, "HubbleParam", 1.0); attr_d(hdr, "Omega0", 0.0); attr_d(hdr, "OmegaLambda", 0.0);
    H5Gclose(hdr);

    hid_t g = H5Gcreate2(f, "PartType0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    auto write3 = [&](const char* name, const std::vector<double>& a,
                      const std::vector<double>& b, const std::vector<double>& c) {
        hsize_t dims[2] = {n, 3};
        hid_t s = H5Screate_simple(2, dims, nullptr);
        hid_t d = H5Dcreate2(g, name, H5T_NATIVE_DOUBLE, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        std::vector<double> buf(3*n);
        for (size_t i = 0; i < n; ++i) { buf[3*i]=a[i]; buf[3*i+1]=b[i]; buf[3*i+2]=c[i]; }
        H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        H5Dclose(d); H5Sclose(s);
    };
    auto write1 = [&](const char* name, const std::vector<double>& a) {
        hsize_t dims = n;
        hid_t s = H5Screate_simple(1, &dims, nullptr);
        hid_t d = H5Dcreate2(g, name, H5T_NATIVE_DOUBLE, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.data());
        H5Dclose(d); H5Sclose(s);
    };
    write3("Coordinates", S.P.x, S.P.y, S.P.z);
    write3("Velocities", S.vx, S.vy, S.vz);
    write1("Masses", S.P.m);
    write1("InternalEnergy", S.u);
    write1("Density", S.rho);
    write1("SmoothingLength", S.h);
    {
        hsize_t dims = n;
        hid_t s = H5Screate_simple(1, &dims, nullptr);
        hid_t d = H5Dcreate2(g, "ParticleIDs", H5T_NATIVE_LLONG, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(d, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, ids.data());
        H5Dclose(d); H5Sclose(s);
    }
    H5Gclose(g); H5Fclose(f);
    printf("wrote %s (t=%.6g)\n", fn, time); fflush(stdout);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Idle ranks EXIT, they do not wait: OpenMPI barriers busy-wait, so parking N-1 ranks at a
    // barrier spins N-1 cores at 100% for the whole run -- the exact __sched_yield pathology this
    // engine exists to remove. mpirun is fine with ranks finalizing at different times.
    if (rank != 0) { MPI_Finalize(); return 0; }
    if (argc < 2) { fprintf(stderr, "usage: %s <paramsfile> [restartflag]\n", argv[0]); return 2; }

    auto kv = parse_kv(argv[1]);
    auto need = [&](const char* k)->std::string {
        auto it = kv.find(k);
        if (it == kv.end()) { fprintf(stderr, "params missing %s\n", k); exit(1); }
        return it->second;
    };
    std::string icfile = need("InitCondFile") + ".hdf5";
    std::string outdir = kv.count("OutputDir") ? kv["OutputDir"] : "output";
    double tmax   = atof(need("TimeMax").c_str());
    double dtsnap = kv.count("TimeBetSnapshot") ? atof(kv["TimeBetSnapshot"].c_str()) : tmax;
    double dtmax  = kv.count("MaxSizeTimestep") ? atof(kv["MaxSizeTimestep"].c_str()) : 1e30;
    double desngb = kv.count("DesNumNgb") ? atof(kv["DesNumNgb"].c_str()) : 32.0;
    double cfl    = kv.count("CourantFac") ? atof(kv["CourantFac"].c_str()) : 0.2;
    double box    = kv.count("BoxSize") ? atof(kv["BoxSize"].c_str()) : 0.0;

    int dim = 3; double gamma = 5.0/3.0;
    parse_config(dim, gamma);
    printf("shmem-GIZMO: %s  dim=%d gamma=%.6f box=%g TimeMax=%g DesNumNgb=%g CFL=%g\n",
           icfile.c_str(), dim, gamma, box, tmax, desngb, cfl);
    fflush(stdout);

    // load ICs
    hid_t f = H5Fopen(icfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) { fprintf(stderr, "cannot open %s\n", icfile.c_str()); return 1; }
    hid_t g = H5Gopen2(f, "PartType0", H5P_DEFAULT);
    Sim S;
    S.dim = dim; S.gamma = gamma; S.box = box; S.des_ngb = desngb; S.cfl = cfl;
    S.P.x = h5_read(g, "Coordinates", 0);
    S.P.y = h5_read(g, "Coordinates", 1);
    S.P.z = h5_read(g, "Coordinates", 2);
    S.P.m = h5_read(g, "Masses", -1);
    S.vx = h5_read(g, "Velocities", 0);
    S.vy = h5_read(g, "Velocities", 1);
    S.vz = h5_read(g, "Velocities", 2);
    S.u  = h5_read(g, "InternalEnergy", -1);
    std::vector<double> ids_d = h5_read(g, "ParticleIDs", -1);
    H5Gclose(g); H5Fclose(f);
    S.P.soft.assign(S.size(), 0.0);
    std::vector<long long> ids(S.size());
    for (size_t i = 0; i < S.size(); ++i) ids[i] = (long long)ids_d[i];

    (void)system(("mkdir -p " + outdir).c_str());

    // snapshot 0 needs Density/h populated: run the volume solve once without stepping
    {
        Tree T = build(S.P);
        std::vector<uint32_t> all(S.size());
        for (size_t i = 0; i < all.size(); ++i) all[i] = (uint32_t)i;
        DensityResult R = density(T, S.P, all, S.des_ngb, {}, S.box, S.dim);
        S.h = R.h;
        S.rho.assign(S.size(), 0.0);
        for (size_t i = 0; i < S.size(); ++i) S.rho[i] = R.rho[i];
    }
    write_snapshot(S, ids, outdir, 0, 0.0);

    double t = 0; int snap = 1; int steps = 0;
    double next_snap = dtsnap;
    while (t < tmax - 1e-12) {
        double target = std::min(next_snap, tmax);
        t += mfm_step(S, std::min(dtmax, target - t));
        ++steps;
        if (t >= target - 1e-12 && target < tmax) {
            write_snapshot(S, ids, outdir, snap++, t);
            next_snap += dtsnap;
        }
    }
    write_snapshot(S, ids, outdir, snap, t);
    printf("done: t=%.6g in %d steps\n", t, steps);
    MPI_Finalize();
    return 0;
}
