# Makefile for the High Performance Computing programming project,
# Academic Year 2022/2023.
#
# Available targets:
#
# - sph: builds the non-GUI version (default)
#
# - sph.gui: builds the GUI version
#
# - all: builds both the GUI and non-GUI versions
#
# - clean: clean up
#
# Last modified on 2022-11-27 by Moreno Marzolla

EXE:=sph sph.gui omp-sph omp-sph.gui
CFLAGS+=-std=c99 -Wall -Wpedantic
LDLIBS=-lm

.PHONY: clean

sph: sph.c

omp-sph: omp-sph.c
	gcc -fopenmp -std=c99 -Wall -Wpedantic omp-sph.c -o omp-sph -lm
omp-sph.gui: omp-sph.c
	gcc -fopenmp -std=c99 -Wall -Wpedantic -DGUI omp-sph.c -lm -lglut -lGL -lX11 -o omp-sph.gui

mpi-sph: mpi-sph.c
	mpicc -std=c99 -Wall -Wpedantic -o mpi-sph mpi-sph.c -lm

gui: sph.gui

all: $(EXE)

sph.gui: CFLAGS+=-DGUI
sph.gui: LDLIBS+=-lglut -lGL -lX11
sph.gui: sph.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

clean:
	\rm -f $(EXE) *.o *~
