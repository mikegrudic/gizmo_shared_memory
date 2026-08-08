# Standalone build of the shared-memory prototype. Not yet wired into GIZMO's main Makefile:
# the tree and threading layer are developed and validated against brute force here first, and
# replace forcetree/domain once the hydro loop passes the tier-1 tests.
CXX      ?= g++
CXXFLAGS ?= -O2 -fopenmp -march=native -Wall
BIN = test_tree bench_layout bench_postsink step_overhead
all: $(BIN)
test_tree: test_tree.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ test_tree.cc tree.cc
bench_layout: bench_layout.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ bench_layout.cc tree.cc
bench_postsink: bench_postsink.cc tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ bench_postsink.cc tree.cc
step_overhead: step_overhead.cc
	$(CXX) $(CXXFLAGS) -o $@ step_overhead.cc
clean:
	rm -f $(BIN)
.PHONY: all clean
test_hydro: test_hydro.cc hydro.cc hydro.h tree.cc tree.h
	$(CXX) $(CXXFLAGS) -o $@ test_hydro.cc hydro.cc tree.cc
