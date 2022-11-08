/****************************************************************************
 *
 * mpi-ring.c - Ring communication with MPI
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% HPC - Ring communication with MPI
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-22

Write an MPI program [mpi-ring.c](mpi-ring.c) which implements a ring
communication between the processes. In details, let $P$ be the number
of MPI processes (to be specified with the `mpirun` command); then,
the program should behave according to the following specification:

- The program receives an integer $K$ from the command line. $K$
  represents the number of "turns" of the ring that ($K \geq
  1$). Remember that all MPI processes have access to the command
  line, so that each one knows the value $K$ without the need to
  communicate.

- Process 0 (the master) sends process 1 an integer, whose
  initial value is 1.

- Every process $p$ (including the master) receives a value $v$ from
  the predecessor $p-1$, and sends $(v + 1)$ to the successor
  $p+1$. The processes are considered to be organized as a ring, so
  that the predecessor of 0 is $(P - 1)$, and the successor of $(P -
  1)$ is 0.

- The master prints the value received after the $K$-th iteration and
  the program terminates. Given the number $P$ of processes and the
  value of $K$, what value should be printed at the end by the master?

For example, if $K = 2$ and there are $P = 4$ processes, the
communication should be as shown in Figure 1 (circles are MPI
processes; arrows are messages whose content is the number shown
above). There are $K = 2$ "turns" of the ring; at the end, process 0
receives and prints the value 8.

![Figure 1: Ring communication](mpi-ring.png)

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-ring.c -o mpi-ring

To execute:

        mpirun -n 4 ./mpi-ring

## Files

- [mpi-ring.c](mpi-ring.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include </usr/lib/aarch64-linux-gnu/openmpi/include/mpi.h>

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, K = 10;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        K = atoi(argv[1]);
    }
    /* [TODO] Rest of the code here... */

    MPI_Finalize();
    return EXIT_SUCCESS;
}
