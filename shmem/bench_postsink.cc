// Tree walk timed on the REAL post-sink state, not a synthetic proxy.
//
// Reads snapshot_052 of the BBB03 run (3.49e6 cells, 1 sink, Time=0.2014) and takes the active set
// to be the DENSEST cells, since dt ~ 1/sqrt(G rho) orders timesteps by density. That set is what
// the deepest timebin actually contains: for 620 cells it spans 0.0033 pc, under 1% of the cloud.
//
// Compared against the same number of cells chosen at RANDOM. That is the pessimal case, and it is
// what a synthetic active set gives you by default; the two are reported side by side because the
// gap between them is large enough to invalidate any walk timing measured the naive way.
#include "tree.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
using namespace shmem;
using clk=std::chrono::steady_clock;
static double ms(clk::time_point a){return std::chrono::duration<double,std::milli>(clk::now()-a).count();}

int main(int argc,char**argv){
    const char* path = (argc>1)?argv[1]:"/tmp/claude-8d47/postsink.bin";
    FILE* fh=fopen(path,"rb"); if(!fh){printf("  cannot open %s\n",path);return 1;}
    uint64_t N; if(fread(&N,8,1,fh)!=1){printf("  short read\n");return 1;}
    Particles P; P.x.resize(N);P.y.resize(N);P.z.resize(N);P.m.resize(N);P.soft.resize(N);
    for(auto* v:{&P.x,&P.y,&P.z,&P.m,&P.soft}) if(fread(v->data(),8,N,fh)!=N){printf("  short read\n");return 1;}
    std::vector<uint32_t> bydens(N); if(fread(bydens.data(),4,N,fh)!=N){printf("  short read\n");return 1;}
    fclose(fh);

    auto t0=clk::now(); Tree T=build(P); double tb=ms(t0);
    printf("  real post-sink state: N=%llu, tree %zu nodes, build %.0f ms, %d threads\n",
           (unsigned long long)N, T.nnodes(), tb, omp_get_max_threads());
    std::vector<uint32_t> pos(N); for(uint64_t i=0;i<N;++i) pos[T.orderbuf[i]]=(uint32_t)i;

    printf("\n  %-8s %12s %12s %10s %14s\n","nactive","random ms","densest ms","speedup","us/particle");
    std::vector<double> ax,ay,az;
    for(size_t nact : {(size_t)620,(size_t)6200,(size_t)62000}){
        std::vector<uint32_t> rnd(nact); std::mt19937_64 r2(3);
        for(size_t i=0;i<nact;++i) rnd[i]=(uint32_t)(r2()%N);
        std::vector<uint32_t> dns(bydens.begin(),bydens.begin()+nact);
        std::sort(dns.begin(),dns.end(),[&](uint32_t a,uint32_t b){return pos[a]<pos[b];});
        double br=1e30,bd=1e30;
        accel(T,P,rnd,0.5,1.0,ax,ay,az); accel(T,P,dns,0.5,1.0,ax,ay,az);
        for(int i=0;i<12;++i){
            auto t=clk::now(); accel(T,P,rnd,0.5,1.0,ax,ay,az); br=std::min(br,ms(t));
            t=clk::now();      accel(T,P,dns,0.5,1.0,ax,ay,az); bd=std::min(bd,ms(t));
        }
        printf("  %-8zu %12.3f %12.3f %9.1fx %14.3f\n",nact,br,bd,br/bd,bd*1000/nact);
    }
    return 0;
}
