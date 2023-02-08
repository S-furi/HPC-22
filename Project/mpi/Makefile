GCC=mpicc
CFLAGS=-std=c99 -Wall -Wpedantic
LIBS=-lm

all: mpi-sph


mpi-sph: mpi-sph.c
	${GCC} ${CFLAGS} -o mpi-sph mpi-sph.c ${LIBS}


.PHONY: clean

clean:
	rm mpi-sph
