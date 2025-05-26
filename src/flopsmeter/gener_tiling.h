/*
** gener_tiling.h
**
** generate cache-optimal tiling for matrix multiplication
**
** Copyright by Maulis, Adam, 2025 in terms of AGPL v3 or newer
**
*/

#ifndef _GENER_TILING_H
#define _GENER_TILING_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



#define GENERSTATE_MAGIC 4294967189u /*a nice prime, no other meaning */


struct generstate{
    double opimal_tiling_size;
    size_t granularity[2];
    size_t base_tilesize[2];      // in terms of granularity
    size_t tilesize_remainder[2]; // in terms of granularity
    size_t current_tilenum[2];    // loop variable
    size_t tilenumber[2];         // loop delimiter
    int row_increment;            // or decrement: +1 or -1
    uint32_t magic;
};

int gener_tiling_init(struct generstate * gs, size_t * matrix_size, size_t K, size_t cachesize, size_t * granularity);
int gener_tiling_next(struct generstate * gs, size_t * offset, size_t * tilesize);

#endif /* _GENER_TILING_H */
