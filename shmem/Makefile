# Standalone build of the shared-memory prototype. Not yet wired into GIZMO's main Makefile:
# the tree and threading layer are developed and validated against brute force here first, and
# replace forcetree/domain once the hydro loop passes the tier-1 tests.
CXX      ?= g++
# C++17 is the floor the site compiler (gcc 8.5) supports fully: structured bindings, if constexpr,
# std::array CTAD and fold expressions are all used below. No std::span -- that needs gcc 10.
CXXFLAGS ?= -O2 -fopenmp -march=native -Wall -std=c++17
# Optional: cache each active particle's neighbour list for the step, so the volume/zeta,
# gradient and flux phases share ONE tree traversal instead of repeating the same search three
# times. Trades memory (~4 bytes per neighbour per active particle, plus a transient second copy
# while the per-thread buffers merge) for traversals. Off by default; build with
#   make GIZMO CACHE_NEIGHBORS=1
CACHE_NEIGHBORS ?= 0
ifeq ($(CACHE_NEIGHBORS),1)
CXXFLAGS += -DSHMEM_CACHE_NEIGHBORS
endif
BIN = test_tree bench_layout bench_postsink step_overhead
all: $(BIN)
test_tree: test_tree.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ test_tree.cc tree.cc
bench_layout: bench_layout.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ bench_layout.cc tree.cc
bench_postsink: bench_postsink.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ bench_postsink.cc tree.cc
bench_treeforce: bench_treeforce.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ bench_treeforce.cc tree.cc
bench_activeset: bench_activeset.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ bench_activeset.cc tree.cc
bench_nodegeom: bench_nodegeom.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ bench_nodegeom.cc tree.cc
bench_density: bench_density.cc hydro.cc hydro.h tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ bench_density.cc hydro.cc tree.cc
step_overhead: step_overhead.cc
	$(CXX) $(CXXFLAGS) -o $@ step_overhead.cc
clean:
	rm -f $(BIN)
.PHONY: all clean
test_hydro: test_hydro.cc hydro.cc hydro.h tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ test_hydro.cc hydro.cc tree.cc
test_mfm: test_mfm.cc mfm.cc mfm.h hydro.cc hydro.h tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ test_mfm.cc mfm.cc hydro.cc tree.cc
suite_runner: suite_runner.cc mfm.cc mfm.h hydro.cc hydro.h tree.cc tree.h
	$(CXX) $(CXXFLAGS) -I$(HDF5_BASE)/include -o $@ suite_runner.cc mfm.cc hydro.cc tree.cc -L$(HDF5_BASE)/lib -lhdf5 -Wl,-rpath,$(HDF5_BASE)/lib
GIZMO: gizmo_main.cc mfm.cc mfm.h hydro.cc hydro.h tree.cc tree.h
	mpicxx $(CXXFLAGS) -I$(HDF5_BASE)/include -o $@ gizmo_main.cc mfm.cc hydro.cc tree.cc -L$(HDF5_BASE)/lib -lhdf5 -Wl,-rpath,$(HDF5_BASE)/lib
