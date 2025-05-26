/*
** gener_tiling_test.c
**
** test for gener_tiling.c
**
** Copyright by Maulis, Adam, 2025 in terms of AGPL v3 or newer
**
*/



#include "gener_tiling.h"

int main(void)
{
    struct generstate mystate;
    int retval;
    size_t matrix_size[2];
    size_t granularity[2];
    size_t offset[2];
    size_t tilesize[2];
    size_t cachesize;
    size_t K;

    cachesize = 40;
    K = 1;

    matrix_size[0] = 18;
    matrix_size[1] = 13;
    granularity[0] = granularity[1] = 2;


    retval = gener_tiling_init( &mystate, matrix_size, K, cachesize, granularity);
    printf("First retval%d \n", retval);
    
    for(retval = gener_tiling_next( &mystate, offset, tilesize); retval != 0; retval = gener_tiling_next( &mystate, offset, tilesize)){
        printf("Loop: offset:(%lu, %lu) tilesize:(%lu, %lu)\n", offset[0], offset[1], tilesize[0], tilesize[1]);
    } /* end for gener_tiling_next() */
    return 0;
}
