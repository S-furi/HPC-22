/***************************************************************************
 *
 * simd-cat-map.c - Arnold's cat map
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
% HPC - La mappa del gatto di Arnold
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2022-11-26

![](cat-map.png)

Scopo di questo esercizio è sviluppare una versione SIMD di una
funzione che calcola l'iterata della _mappa del gatto di Arnold_, una
vecchia conoscenza che abbiamo già incontrato in altre esercitazioni.
Riportiamo nel seguito la descrizione del problema.

La mappa del gatto trasforma una immagine $P$ di dimensione $N \times
N$ in una nuova immagine $P'$ delle stesse dimensioni. Per ogni $0
\leq x < N,\ 0 \leq y < N$, il pixel di coordinate $(x,y)$ in $P$
viene collocato nella posizione $(x',y')$ di $P'$ dove:

$$
x' = (2x + y) \bmod N, \qquad y' = (x + y) \bmod N
$$

("mod" è l'operatore modulo, corrispondente all'operatore `%` del
linguaggio C). Si può assumere che le coordinate $(0, 0)$ indichino il
pixel in alto a sinistra e le coordinate $(N-1, N-1)$ quello in basso
a destra, in modo da poter indicizzare l'immagine come se fosse una
matrice in linguaggio C. La Figura 1 mostra graficamente la
trasformazione.

![Figura 1: La mappa del gatto di Arnold](cat-map.svg)

La mappa del gatto ha proprietà sorprendenti. Applicata ad una
immagine ne produce una versione molto distorta. Applicata nuovamente
a quest'ultima immagine, ne produce una ancora più distorta, e così
via. Tuttavia, dopo un certo numero di iterazioni (il cui valore
dipende da $N$, ma che in ogni caso è sempre minore o uguale a $3N$)
ricompare l'immagine di partenza! (si veda la Figura 2).

![Figura 2: Alcune immagini ottenute iterando la mappa del gatto $k$ volte](cat-map-demo.png)

Il _tempo minimo di ricorrenza_ per l'immagine
[cat1368.pgm](cat1368.pgm) di dimensione $1368 \times 1368$ fornita
come esempio è $36$: iterando $k$ volte della mappa del gatto si
otterrà l'immagine originale se e solo se $k$ è multiplo di 36. Non è
nota alcuna formula analitica che leghi il tempo minimo di ricorrenza
alla dimensione $N$ dell'immagine.

Viene fornito un programma sequenziale che calcola la $k$-esima iterata
della mappa del gatto usando la CPU. Il programma viene invocato
specificando sulla riga di comando il numero di iterazioni $k$. Il
programma legge una immagine in formato PGM da standard input, e
produce una nuova immagine su standard output ottenuta applicando $k$
volte la mappa del gatto. Occorre ricordarsi di redirezionare lo
standard output su un file, come indicato nelle istruzioni nel
sorgente.  La struttura della funzione che calcola la k-esima iterata
della mappa del gatto è molto semplice:

```C
for (y=0; y<N; y++) {
	for (x=0; x<N; x++) {
		\/\* calcola le coordinate (xnew, ynew) del punto (x, y)
			dopo k applicazioni della mappa del gatto \*\/
		next[xnew + ynew*N] = cur[x+y*N];
	}
}
```

Per sfruttare il parallelismo SIMD possiamo ragionare come segue:
anziché calcolare le nuove coordinate di un punto alla volta,
calcoliamo le coordinate di quattro punti adiacenti $(x, y)$,
$(x+1,y)$, $(x+2,y)$, $(x+3,y)$ usando i _vector datatype_ del
compilatore. Per fare questo, definiamo le seguenti variabili di tipo
`v4i` (vettori SIMD di 4 interi):

- `vx`, `vy`: coordinate di quattro punti adiacenti, prima
  dell'applicazione della mappa del gatto;

- `vxnew`, `vynew`: nuova coordinate dei punti di cui sopra dopo
  l'applicazione della mappa del gatto.

Ricordiamo che il tipo `v4i` si definisce con `gcc` come

```C
	typedef int v4i __attribute__((vector_size(16)));
	#define VLEN (sizeof(v4i)/sizeof(int))
```

Posto $vx = \{x, x+1, x+2, x+3\}$, $vy = \{y, y, y, y\}$, possiamo
applicare ad essi le stesse operazioni aritmetiche applicate agli
scalari _x_ e _y_ per ottenere le nuove coordinate _vxnew_, _vynew_.
Fatto questo, al posto della singola istruzione:

```C
	next[xnew + ynew*N] = cur[x+y*N];
```

per spostare materialmente i pixel nella nuova posizione occorre
eseguire quattro istruzioni scalari:

```C
	next[vxnew[0] + vynew[0]*N] = cur[vx[0] + vy[0]*N];
	next[vxnew[1] + vynew[1]*N] = cur[vx[1] + vy[1]*N];
	next[vxnew[2] + vynew[2]*N] = cur[vx[2] + vy[2]*N];
	next[vxnew[3] + vynew[3]*N] = cur[vx[3] + vy[3]*n];
```

Si assuma che la dimensione $N$ dell'immagine sia sempre multipla di 4.

Compilare con:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native simd-cat-map.c -o simd-cat-map

Eseguire con:

        ./simd-cat-map k < input_file > output_file

Esempio:

        ./simd-cat-map 100 < cat368.pgm > cat1368-100.pgm

## Estensione

Le prestazioni della versione SIMD dell'algoritmo della mappa del
gatto dovrebbero risultare solo marginalmente migliori della versione
scalare (potrebbero addirittura essere peggiori). Analizzando il
codice assembly prodotto dal compilatore, si scopre che il calcolo del
modulo nelle due espressioni

```C
	vxnew = (2*vxold+vyold) % N;
	vynew = (vxold + vyold) % N;
```

viene realizzato usando operazioni scalari. Consultando la lista dei
_SIMD intrinsics_ sul [sito di
Intel](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
si scopre che non esiste una istruzione SIMD che realizzi la divisione
intera. Per migliorare le prestazioni del programma occorre quindi
ingegnarsi per calcolare i moduli senza fare uso della
divisione. Ragionando in termini scalari, osserviamo che se $0 \leq
xold < N$ e $0 \leq yold < N$, allora si ha necessariamente che $0
\leq 2 \times xold + yold < 3N$ e $0 \leq xold+yold < 2N$.

Pertanto, sempre in termini scalari, possiamo realizzare il calcolo di
`xnew` e `ynew` come segue:

```C
	xnew = (2*xold + yold);
	if (xnew >= N) { xnew = xnew - N; }
	if (xnew >= N) { xnew = xnew - N; }
	ynew = (xold + yold);
	if (ynew >= N) { ynew = ynew - N; }
```

Il codice precedente è meno leggibile della versione che usa
l'operatore modulo, ma ha il vantaggio di poter essere vettorizzato
ricorrendo al meccanismo di "selection and masking" visto a
lezione. Ad esempio, l'istruzione

```C
	if (xnew >= N) { xnew = xnew - N; }
```

può essere riscritta come

```C
	const v4i mask = (xnew >= N);
	xnew = (mask & (xnew - N)) | (mask & xnew);
```

che può essere ulteriormente semplificata come:

```C
	const v4i mask = (xnew >= N);
	xnew = xnew - (mask & N);
```

Si ottiene in questo modo un programma più complesso della versione
scalare, ma più veloce in quanto si riesce a sfruttare al meglio le
istruzioni SIMD.

Per compilare:

        gcc -std=c99 -Wall -Wpedantic -march=native -O2 simd-cat-map.c -o simd-cat-map

Per eseguire:

        ./simd-cat-map [niter] < in.pgm > out.pgm

Esempio:

        ./simd-cat-map 1024 < cat1368.pgm > cat1368-1024.pgm

## File

- [simd-cat-map.c](simd-cat-map.c)
- [hpc.h](hpc.h)
- [cat1368.pgm](cat1368.pgm) (il tempo di ricorrenza di questa immagine è 36)

 ***/

/* The following #define is required by posix_memalign(). It MUST
   be defined before including any other files. */
#define _XOPEN_SOURCE 600

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef int v4i __attribute__((vector_size(16)));
#define VLEN (sizeof(v4i)/sizeof(int))

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
    /* The pointer img->bmap must be properly aligned to allow aligned
       SIMD load/stores to work. */
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

void cat_map( PGM_image* img, int k )
{
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next;
    int x, y, i, ret;

    assert( img->width == img->height );
    assert( img->width % VLEN == 0);

    ret = posix_memalign((void**)&next, __BIGGEST_ALIGNMENT__, N*N*sizeof(*next));
    assert( 0 == ret );

    for (y=0; y<N; y++) {
        v4i vx = {0, 1, 2, 3};
        v4i vy = {y, y, y, y};
        for (x=0; x<N - VLEN + 1; x += VLEN) {
            v4i xold = vx, xnew = xold;
            v4i yold = vy, ynew = yold;
            for (i=0; i<k; i++) {
#if 0
              xnew = (2*xold+yold) % N;
              ynew = (xold + yold) % N;
#else
              v4i mask;

              xnew = (2*xold+yold);
              mask = (xnew >= N);
              xnew = (mask & (xnew - N)) | (~mask & xnew);
              mask = (xnew >= N);
              xnew = (mask & (xnew - N)) | (~mask & xnew);

              ynew = (xold + yold);
              mask = (ynew >= N);
              ynew = (mask & (ynew - N)) | (~mask & ynew);
#endif
              xold = xnew;
              yold = ynew;
            }
            /* next[xnew + ynew*N] = cur[x+y*N]; */
            next[xnew[0] + ynew[0]*N] = cur[vx[0]+vy[0]*N];
            next[xnew[1] + ynew[1]*N] = cur[vx[1]+vy[1]*N];
            next[xnew[2] + ynew[2]*N] = cur[vx[2]+vy[2]*N];
            next[xnew[3] + ynew[3]*N] = cur[vx[3]+vy[3]*N];
        }
    }

    img->bmap = next;
    free(cur);
}

int main( int argc, char* argv[] )
{
    PGM_image img;
    int niter;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter < in.pgm > out.pgm\n\nExample: %s 684 < cat1368.pgm > out1368.pgm\n", argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);
    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }
    if ( img.width % VLEN ) {
        fprintf(stderr, "FATAL: this program expects the image width (%d) to be a multiple of %d\n", img.width, (int)VLEN);
        return EXIT_FAILURE;
    }
    const double tstart = hpc_gettime();
    cat_map(&img, niter);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "      SIMD width : %d bytes\n", (int)VLEN);
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "     Mpixels/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

    write_pgm(stdout, &img, "produced by simd-cat-map.c");
    free_pgm(&img);
    return EXIT_SUCCESS;
}
