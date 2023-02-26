# Makefile for the High Performance Computing programming project,
# Academic Year 2022/2023.
#
# Available targets:
#
# - omp-sph: builds the parallel version using OpenMP
#
# - sph.gui: builds the parallel version using MPI
#
# - all: builds both the MPI and OpenMP versions
#
# - clean: clean up

CFLAGS=-std=c99 -Wall -Wpedantic
LIBS=-fopenmp -lm

all: omp-sph mpi-sph


omp-sph: omp-sph.c
	gcc ${CFLAGS} -o omp-sph omp-sph.c ${LIBS}

mpi-sph: mpi-sph.c
	mpicc ${CFLAGS} -o mpi-sph mpi-sph.c -lm


.PHONY: clean

clean:
	rm mpi-sph omp-sph
