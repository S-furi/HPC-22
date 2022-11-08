/****************************************************************************
 *
 * omp-merge-sort.c - Merge Sort with OpenMP tasks
 *
 * Copyright (C) 2017--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Merge Sort with OpenMP tasks
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-08-14

The file [omp-merge-sort.c](omp-merge-sort.c) contains an
implementation of the recursive _Merge Sort_ algorithm. The
implementation reverts to _Selection Sort_ when the size of the
subvector becomes lower than a cutoff value. This is a standard
optimization that tries to avoid the overhead of recursive calls on
very small vectors.

The program generates a random permutation of the integers $0, 1,
\ldots, n-1$, and then sorts the permutation using Merge Sort. It if
therefore easy to check the correctness of the result.

The goal of this exercise is to parallelize the implementation using
OpenMP tasks; you should create a separate task for each recursive
call. Measure the execution time of the parallel version and compare
the result with the serial implementation. To get meaningful results,
you should chose an input vector that is large enough so that at least
a few seconds are required to sort the vector using all available
processor cores.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-merge-sort.c -o omp-merge-sort

Run with:

        ./omp-merge-sort 50000

## Files

- [omp-merge-sort.c](omp-merge-sort.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

int min(int a, int b)
{
    return (a < b ? a : b);
}

void swap(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * Sort v[low..high] using selection sort. This function will be used
 * for small vectors only. Do not parallelize this.
 */
void selectionsort(int* v, int low, int high)
{
    int i, j;
    for (i=low; i<high; i++) {
        for (j=i+1; j<=high; j++) {
            if (v[i] > v[j]) {
                swap(&v[i], &v[j]);
            }
        }
    }
}

/**
 * Merge src[low..mid] with src[mid+1..high], put the result in
 * dst[low..high].
 *
 * Do not parallelize this function (it could be done, but is very
 * difficult, see
 * http://www.drdobbs.com/parallel/parallel-merge/229204454
 * https://en.wikipedia.org/wiki/Merge_algorithm#Parallel_merge )
 */
void merge(int* src, int low, int mid, int high, int* dst)
{
    int i=low, j=mid+1, k=low;
    while (i<=mid && j<=high) {
        if (src[i] <= src[j]) {
            dst[k] = src[i++];
        } else {
            dst[k] = src[j++];
        }
        k++;
    }
    /* Handle leftovers */
    while (i<=mid) {
        dst[k] = src[i++];
        k++;
    }
    while (j<=high) {
        dst[k] = src[j++];
        k++;
    }
}

/**
 * Sort v[i..j] using the recursive version of Merge Sort; the array
 * tmp[i..j] is used as a temporary buffer (the caller is responsible
 * for providing a suitably sized array tmp).
 */
void mergesort_rec(int* v, int i, int j, int* tmp)
{
    const int CUTOFF = 64;
    /* If the portion to be sorted is smaller than the CUTOFF, use
       selectoin sort. This is a widely used optimization that limits
       the overhead of recursion for small vectors. The optimal value
       of the cutoff is system-dependent; the value used here is just
       an example. */
    if ( j - i + 1 < CUTOFF )
        selectionsort(v, i, j);
    else {
        const int m = (i+j)/2;
        /* [TODO] The following two recursive invocation of
           mergesort_rec() are independent, and therefore can run in
           parallel. Create two OpenMP tasks to sort the first and
           second half of the array; then, wait for all tasks to
           complete before merging the results. */
#pragma omp task shared(i, m)
        mergesort_rec(v, i, m, tmp);
#pragma omp task shared(j, m)
        mergesort_rec(v, m+1, j, tmp);
        /* When using OpenMP, we must wait here for the recursive
           invocations of mergesort_rec() to terminate before merging
           the result */
#pragma omp taskwait
        merge(v, i, m, j, tmp);
        /* copy the sorted data back to v */
        memcpy(v+i, tmp+i, (j-i+1)*sizeof(v[0]));
    }
}

/**
 * Sort v[] of length n using Merge Sort; after allocating a temporary
 * array with the same size of a (used for merging), this function
 * just calls mergesort_rec with the appropriate parameters.  After
 * mergesort_rec terminates, the temporary array is deallocated.
 */
void mergesort(int *v, int n)
{
    int* tmp = (int*)malloc(n*sizeof(v[0]));
    assert(tmp != NULL);
    /* [TODO] Parallelize the body of this function. You should create
       a pool of thread here, and ensure that only one thread calls
       mergesort_rec() to start the recursion. */
#pragma omp parallel default(none) shared(n, v, tmp)
    {
#pragma omp master
      mergesort_rec(v, 0, n-1, tmp);
    }
    free(tmp);
}

/* Returns a random integer in the range [a..b], inclusive */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/**
 * Fills a[] with a random permutation of the intergers 0..n-1; the
 * caller is responsible for allocating a
 */
void fill(int* a, int n)
{
    int i;
    for (i=0; i<n; i++) {
        a[i] = (int)i;
    }
    for (i=0; i<n-1; i++) {
        int j = randab(i, n-1);
        swap(a+i, a+j);
    }
}

/* Return 1 iff a[] contains the values 0, 1, ... n-1, in that order */
int check(const int* a, int n)
{
    int i;
    for (i=0; i<n; i++) {
        if ( a[i] != i ) {
            fprintf(stderr, "Expected a[%d]=%d, got %d\n", i, i, a[i]);
            return 0;
        }
    }
    return 1;
}

int main( int argc, char* argv[] )
{
    int n = 10000000;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if (n > 1000000000) {
        fprintf(stderr, "FATAL: array too large\n");
        return EXIT_FAILURE;
    }

    int *a = (int*)malloc(n*sizeof(a[0]));
    assert(a != NULL);

    printf("Initializing array...\n");
    fill(a, n);
    printf("Sorting %d elements...", n); fflush(stdout);
    const double tstart = omp_get_wtime();
    mergesort(a, n);
    const double elapsed = omp_get_wtime() - tstart;
    printf("done\n");
    const int ok = check(a, n);
    printf("Check %s\n", (ok ? "OK" : "failed"));
    printf("Elapsed time: %f\n", elapsed);

    free(a);

    return EXIT_SUCCESS;
}
