/****************************************************************************
 *
 * cuda-cat-map.cu - Arnold's cat map
 *
 * Copyright (C) 2016--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Arnold's cat map
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-23

![Vladimir Igorevich Arnold (1937--2010). By Svetlana Tretyakova - <http://www.mccme.ru/arnold/pool/original/VI_Arnold-05.jpg>, CC BY-SA 3.0](Vladimir_Arnold.jpg)

[Arnold's cat map](https://en.wikipedia.org/wiki/Arnold%27s_cat_map)
is a continuous chaotic function that has been studied in the '60s by
the Russian mathematician [Vladimir Igorevich
Arnold](https://en.wikipedia.org/wiki/Vladimir_Arnold). In its
discrete version, the function can be understood as a transformation
of a bitmapped image $P$ of size $N \times N$ into a new image $P'$ of
the same size. For each $0 \leq x, y < N$, the pixel of coordinates
$(x,y)$ in $P$ is mapped into a new position $C(x, y) = (x', y')$ in
$P'$ where

$$
x' = (2x + y) \bmod N, \qquad y' = (x + y) \bmod N
$$

("mod" is the integer remainder operator, i.e., operator `%` of the C
language). We may assume that $(0, 0)$ is top left and $(N-1, N-1)$
bottom right, so that the bitmap can be encoded as a regular
two-dimensional C matrix.

The transformation corresponds to a linear "stretching" of the image,
that is then broken down into triangles that are rearranged as shown
in Figure 1.

![Figura 1: Arnold's cat map](cat-map.svg)

Arnold's cat map has interesting properties. Let $C^k(x, y)$ be the
result of iterating $k$ times the function $C$, i.e.:

$$
C^k(x, y) = \begin{cases}
(x, y) & \mbox{if $k=0$}\\
C(C^{k-1}(x,y)) & \mbox{if $k>0$}
\end{cases}
$$

Therefore, $C^2(x,y) = C(C(x,y))$, $C^3(x,y) = C(C(C(x,y)))$, and so
on.

If we take an image and apply $C$ once, we get a severely distorted
version of the input. If we apply $C$ on the resulting image, we get
an even more distorted image. As we keep applying $C$, the original
image is no longer discernible. However, after a certain number of
iterations that depend on $N$ and has been proved to never exceed
$3N$, we get back the original image! (Figure 2).

![Figura 2: Some iterations of the cat map](cat-map-demo.png)

The _minimum recurrence time_ for an image is the minimum positive
integer $k \geq 1$ such that $C^k(x, y) = (x, y)$ for all $(x, y)$. In
simple terms, the minimum recurrence time is the minimum number of
iterations of the cat map that produce the starting image.

For example, the minimum recurrence time for
[cat1368.pgm](cat1368.pgm) of size $1368 \times 1368$ is $36$. As said
before, the minimum recurrence time depends on the image size $N$.
Unfortunately, no closed formula is known to compute the minimum
recurrence time as a function of $N$, although there are results and
bounds that apply to specific cases.

You are provided with a serial program that computes the $k$-th
iterate of Arnold's cat map on a square image. The program reads the
input from standard input in
[PGM](https://en.wikipedia.org/wiki/Netpbm) (_Portable GrayMap_)
format. The results is printed to standard output in PGM format. For
example:

        ./cuda-cat-map 100 < cat1368.pgm > cat1368-100.pgm

applies the cat map $k=100$ times on `cat1368.phm` and saves the
result to `cat1368-100.pgm`.

To display a PGM image you might need to convert it to a different
format, e.g., JPEG. Under Linux you can use `convert` from the
[ImageMagick](https://imagemagick.org/) package:

        convert cat1368-100.pgm cat1368-100.jpeg

To make use of CUDA parallelism, define a 2D grid of 2D blocks that
covers the input image. The block size is $\mathit{BLKDIM} \times
\mathit{BLKDIM}$, with `BLKDIM = 32`, and the grid size is:

$$
(N + \mathit{BLKDIM} – 1) / \mathit{BLKDIM} \times (N + \mathit{BLKDIM} – 1) / \mathit{BLKDIM}
$$

Each thread applies a single iteration of the cat map and copies one
pixel from the input image to the correct position of the output
image.  The kernel has the following signature:

```C
__global__ void cat_map_iter( unsigned char *cur, unsigned char *next, int N )
```

where $N$ is the height/width of the image. The program must work
correctly even if $N$ is not an integer multiple of _BLKDIM_. Each
thread is mapped to the coordinates $(x, y)$ of a pixel using the
usual formulas:

```C
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
```

Therefore, to compute the $k$-th iteration of the cat map we need to
execute the kernel $k$ times.

A better approach is to define a kernel

```C
__global__ void cat_map_iter_k( unsigned char *cur, unsigned char *next, int N, int k )
```

that applies $k$ iterations of the cat map to the current image.  This
kernel needs to be executed only once, and this saves some significant
overhead associated to kernel calls. The new kernel can be
defined as follows:

```C
const int x = ...;
const int y = ...;
int xcur = x, ycur = y, xnext, ynext;

if ( x < N && y < N ) {
	while (k--) {
		xnext = (2*xcur + ycur) % N;
		ynext = (xcur + ycur) % N;
		xcur = xnext;
		ycur = ynext;
	}
	\/\* copy the pixel (x, y) from the current image to
	the position (xnext, ynext) of the new image \*\/
}
```

I suggest to implement both solutions (the one where the kernel is
executed $k$ times, and the one where the kernel is executed only
once) and measure the execution times to see the difference.

To compile:

        nvcc cuda-cat-map.cu -o cuda-cat-map

To execute:

        ./cuda-cat-map k < input_file > output_file

Example:

        ./cuda-cat-map 100 < cat1368.pgm > cat1368.100.pgm

## Files

- [cuda-cat-map.cu](cuda-cat-map.cu)
- [hpc.h](hpc.h)
- [cat1368.pgm](cat1368.pgm) (the minimum recurrence time of this image is 36)

***/

#include "../hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define BLKDIM 32

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    unsigned char *bmap; /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

const unsigned char WHITE = 255;
const unsigned char BLACK = 0;

/**
 * Initialize a PGM_image object: allocate space for a bitmap of size
 * `width` x `height`, and set all pixels to color `col`
 */
void init_pgm( PGM_image *img, int width, int height, unsigned char col )
{
    int i, j;

    assert(img != NULL);

    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char*)malloc(width*height);
    assert(img->bmap != NULL);
    for (i=0; i<height; i++) {
        for (j=0; j<width; j++) {
            img->bmap[i*width + j] = col;
        }
    }
}

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done.
 */
void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
#if _XOPEN_SOURCE < 600
    img->bmap = (unsigned char*)malloc((img->width)*(img->height)*sizeof(unsigned char));
#else
    /* The pointer img->bmap must be properly aligned to allow SIMD
       instructions, because the compiler emits SIMD instructions for
       aligned load/stores only. */
    int ret = posix_memalign((void**)&(img->bmap), __BIGGEST_ALIGNMENT__, (img->width)*(img->height));
    assert( 0 == ret );
#endif
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    nread = fread(img->bmap, 1, (img->width)*(img->height), f);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write the image `img` to file `f`; if not NULL, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), f);
}

/**
 * Free the bitmap associated with image `img`; note that the
 * structure pointed to by `img` is NOT deallocated; only `img->bmap`
 * is.
 */
void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL; /* not necessary */
    img->width = img->height = img->maxgrey = -1;
}

__global__ void cat_map_iter(unsigned char *cur, unsigned char *next, int N, int k) {
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;

		int xcur = x, ycur = y, xnext, ynext;

		if (x < N && y < N) {
				while(k--) {
						xnext = (2*xcur + ycur) % N;
						ynext = (xcur + ycur) % N;
						xcur = xnext;
						ycur = ynext;
				}
				next[xnext + ynext*N] = cur[x + y*N];
		}
}

/**
 * Compute the `k`-th iterate of the cat map for image `img`. The
 * width and height of the input image must be equal. This function
 * replaces the bitmap of `img` with the one resulting after ierating
 * `k` times the cat map. You need to allocate a temporary image, with
 * the same size of the original one, so that you read the pixel from
 * the "old" image and copy them to the "new" image (this is similar
 * to a stencil computation, as was discussed in class). After
 * applying the cat map to all pixel of the "old" image the role of
 * the two images is exchanged: the "new" image becomes the "old" one,
 * and vice-versa. The temporary image must be deallocated upon exit.
 */
void cat_map( PGM_image* img, int k )
{
    const int N = img->width;
    const size_t size = N * N * sizeof(img->bmap[0]);

		unsigned char *d_cur, *d_next;

		cudaSafeCall(cudaMalloc((void**)&d_cur, size));
		cudaSafeCall(cudaMalloc((void**)&d_next, size));

		cudaSafeCall(cudaMemcpy(d_cur, img->bmap, size, cudaMemcpyHostToDevice));

		dim3 block(BLKDIM, BLKDIM);
		dim3 grid((N+BLKDIM - 1)/BLKDIM, (N+BLKDIM - 1)/BLKDIM);

		cat_map_iter<<<grid, block>>>(d_cur, d_next, N, k);
		cudaCheckError();

		cudaSafeCall(cudaMemcpy(img->bmap, d_next, size, cudaMemcpyDeviceToHost));

		cudaFree(d_cur);
		cudaFree(d_next);

		/*
    assert(next != NULL);
    for (int i=0; i<k; i++) {
        for (int y=0; y<N; y++) {
            for (int x=0; x<N; x++) {
                int xnext = (2*x+y) % N;
                int ynext = (x + y) % N;
                next[xnext + ynext*N] = cur[x+y*N];
            }
        }
        unsigned char *tmp = cur;
        cur = next;
        next = tmp;
    }
    img->bmap = cur;
    free(next);
		*/
}

int main( int argc, char* argv[] )
{
    PGM_image img;
    int niter;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter < input_image > output_image\n", argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);
    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }
    const double tstart = hpc_gettime();
    cat_map(&img, niter);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "     Mpixels/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

    write_pgm(stdout, &img, "produced by cuda-cat-map.cu");
    free_pgm(&img);
    return EXIT_SUCCESS;
}
