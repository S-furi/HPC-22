/****************************************************************************
 *
 * omp-sieve.c -- Sieve of Eratosthenes
 *
 * Copyright (C) 2018--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Sieve of Eratosthenes
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-08-10

The _sieve of Erathostenes_ is an algorithm for identifying the prime
numbers falling within a given range which usually is the set $\{2,
\ldots, n\}$ . A natural number $p \geq 2$ is prime if and only if the
only divisors are 1 and $p$ itself (2 is prime).

To illustrate how the sieve of Eratosthenes works, let us consider the
case $n=20$. We start by listing all integers $2, \ldots n$:

![](omp-sieve1.png)

The first value in the list (2) is prime; we mark all its multiples
and get:

![](omp-sieve2.png)

The next unmarked value (3) is again prime. We mark all its multiples
starting from $3 \times 3$ (indeed, $3 \times 2$ has been
marked at the previous step because it is a multiple of 2). We get:

![](omp-sieve3.png)

The next unmarked value (5) is prime. The smaller unmarked multiple of
5 is $5 \times 5$, because $5 \times 2$, $5 \times 3$ and $5 \times 4$
have all been marked since they are multiples of 2 and 3. However,
since $5 \times 5 > 20$ is outside the upper bound of the interval,
the algorithm terminates and all unmarked numbers are prime:

![](omp-sieve4.png)

The file [omp-sieve.c](omp-sieve.c) contains a serial program that,
given an integer $n \geq 2$, computes the number $\pi(n)$ of primes in
the set $\{2, \ldots n\}$ using the sieve of
Eratosthenes[^1]. Although the serial program could be made more
efficient both in time and space, here it is best to sacrifice
efficiency for readability. The set of unmarked numbers in $\{2,
\ldots, n\}$ is represented by the `isprime[]` array of length $n+1$;
during execution, `isprime[k]` is 0 if and only if $k$ has been
marked, i.e., has been determined to be composite ($2 \leq k \leq n$);
`isprime[0]` e `isprime[1]` are not used.

[^1]: $\pi(n)$ is also called [prime-counting
      function](https://en.wikipedia.org/wiki/Prime-counting_function)

The program contains a function `int mark(char *isprime, int from, int
to, int p)` that marks all multiples of $p$ belonging to the set
$\{\texttt{from}, \ldots \texttt{to}-1\}$. The function returns the
number of values that have been marked for the first time.

The goal is to write a parallel version of the sieve of Erathostenes;
to this aim, you might want to use the following hints.

The main program contains the following instructions:

```C
count = n - 1;
for (i=2; i*i <= n; i++) {
	if (isprime[i]) {
		count -= mark(isprime, i*i, n+1, i);
	}
}
```

To compute $\pi(n)$ we start by initializing `count` as the number of
elements in the set $\{2, \ldots n\}$; every time we mark a value for
the first time, we decrement `count` so that, at the end, we have that
$\pi(n) = \texttt{count}$.

Note that it is not possible to parallelize the _for_ loop above,
because the content of `isprime[]` is possibly modified by function
`mark()`, and this represents a _loop-carried dependency_. However, it
is possible to parallelize the body of function `mark()`. The idea is
to partition the set $[i \times i, n]$ among $P$ OpenMP threads so
that every thread will mark all multiples of $i$ that belong to its
partition.

I suggest that you start using the `omp parallel` construct (not `omp
parallel for`) and compute the bounds of each partition by hand.  It
is not trivial to do so correctly, but this is quite instructive since
during the lectures we only considered the simple case of partitioning
a range $0, \ldots, n-1$, while here the range does not start at zero.

Once you have a working parallel version, you can take the easier
route to use the `omp parallel for` directive and let the compiler
partition the iteration range for you.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-sieve.c -o omp-sieve

Execute with:

        ./omp-sieve [n]

Example:

        OMP_NUM_THREADS=2 ./omp-sieve 1000

As a reference, Table 1 shows the values of $\pi(n)$ for some
$n$. Use the table to check the correctness of your implementation

:Table 1: some values of the prime-counting function $\pi(n)$

          $n$                             $\pi(n)$
-------------  -----------------------------------
            1                                    0
           10                                    4
          100                                   25
         1000                                  168
        10000                                 1229
       100000                                 9592
      1000000                                78498
     10000000                               664579
    100000000                              5761455
   1000000000                             50847534
  10000000000  **Do not try on the server**: uses >10GB of RAM!!
-------------  -----------------------------------

## Files

- [omp-sieve.c](omp-sieve.c)

***/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Mark all mutliples of `p` in the set {`from`, ..., `to`-1}; return how
   many numbers have been marked for the first time. `from` does not
   need to be a multiple of `p`. */
long mark( char *isprime, long from, long to, long p )
{
    long nmarked = 0l;

    /*
     * (from/p) * p, il valore calolato 
     * corrisponde al massimo
     * multiplo di p <= from 
     */

#pragma omp parallel default(none) shared(from, to, p, isprime, nmarked)
    {
      const int my_id = omp_get_thread_num();
      const int num_threads = omp_get_num_threads();
      /* partizionare da from a to (n estremi esclusi) in P unita' d'esecuzione */
      const long n = (to - from);
      long my_from = from + ((n * my_id) / num_threads);
      long my_to = from + ((n * (my_id + 1)) / num_threads);

      my_from = ((my_from + p - 1)/p)*p; /* start from the lowest multiple of p that is >= from */
      for ( long x=my_from; x<my_to; x+=p ) {
          if (isprime[x]) {
              isprime[x] = 0;
#pragma omp atomic
              nmarked++;
          }
      }
    }
    return nmarked;
}

int main( int argc, char *argv[] )
{
    long n = 1000000l, nprimes, i;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atol(argv[1]);
    }

    if (n > (1ul << 31)) {
        fprintf(stderr, "FATAL: n too large\n");
        return EXIT_FAILURE;
    }

    char *isprime = (char*)malloc(n+1); assert(isprime != NULL);
    /* Initially, all numbers are considered primes */
    for (i=0; i<=n; i++)
        isprime[i] = 1;

    nprimes = n-1;
    const double tstart = omp_get_wtime();
    /* main iteration of the sieve */
    for (i=2; i*i <= n; i++) {
        if (isprime[i]) {
            // Loop Carried Dependency
            nprimes -= mark(isprime, i*i, n+1, i);
        }
    }
    const double elapsed = omp_get_wtime() - tstart;
    /* Enable to print the list of primes */
#if 0
    for (i=2; i<=n; i++) {
        if (isprime[i]) {printf("%ld ", i);}
    }
    printf("\n");
#endif
    free(isprime);
    printf("There are %ld primes in {2, ..., %ld}\n", nprimes, n);
    printf("Elapsed time: %f\n", elapsed);
    return EXIT_SUCCESS;
}
