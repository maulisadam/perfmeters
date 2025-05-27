/*
** gener_tiling.c
**
** generate cache-optimal tiling for matrix multiplication
**
** Copyright by Maulis, Adam, 2025 in terms of AGPL v3 or newer
**
**  2dimensional tiling generator for efficient looping.
**
**  instead of:
**      for(tile[0]=0, tile[0] < i_size; tile[0]++)
**          for(tile[1]=0, tile[1] < j_size; tile[1]++)
**              some_code(tile);
**
** use this:
**      struct generstate gs;
**      gener_tiling_init(&gs, ....)
**      for(tile_retval = gener_tiling_next(&gs...tile...); tile_retval != 0; tile_retval = gener_tiling_next(&gs...tile...))
**          some_code(tile);
**
**  Common verbs for this code:
**  OUT = M1 @ M2 where M1 is matrix of size (I, K); M2's size is (K, J); OUT's size is (I,J)
**
**
** What does the cache effinciency mean?
**   We want to divide the problem into tiles of size AxA
**   where the rows/columns of M1 and M2 and the tile of OUT both fit into the cache.
**   2*A*K + A*A smaller or equal than cachesize.
*/

#include "gener_tiling.h"


// example for modulo 5: 0->0; 1->5; 4->5; 5->5; 6->10
__attribute__ ((const)) static inline size_t roundup(size_t what, size_t modulo) // const means side effect-free
{
    return what + (modulo - what % modulo) % modulo;
}

// example for divisor 5: 0->0, 5->1, 6->2, 10->2, 11->3
__attribute__ ((const)) static inline size_t divideup(size_t what, size_t divisor) // const means side effect-free
{
    return (what + divisor - 1)/divisor;
}


/*
**  Return value:
**      1 if firstoffset and tilesize valid;
**      0 if there no tile needed
*/


int gener_tiling_init(
    struct generstate * gs, /* state of this generator */
    size_t * matrix_size,   /* 2-len vector, the (output) matrix size tiling for */
    size_t K,               /* the row length */
    size_t cachesize,       /* in unit of a float/double/number (not a byte) */
    size_t * granularity    /* 2-len vector, eg opencl's local_size  */
    )
{

    if( NULL == gs){
        dprintf(2, "Err: NULL generstate in gener_tiling_init\n");
        exit(2);
    }
    if( GENERSTATE_MAGIC == gs->magic){
        dprintf(2, "Err: already used generstate in gener_tiling_init\n");
        exit(2);
    }
    if( 0 == matrix_size[0] || 0 == matrix_size[1]){
        dprintf(2, "Err: matrix size must be not null!\n");
        exit(2);
    }
    gs->magic = GENERSTATE_MAGIC;

    // We have 2d data set
    // we will tiling for optimal L2 cache usage.
    // let the K be the row length
    // let the C be the cache size
    // the opimal_tiling_size is sqrt(C+K^2)-K in each dimension
    gs->opimal_tiling_size = sqrt( (double)cachesize + (double)(K)*(double)(K)) - (double)(K);
    gs->granularity[0] = granularity[0];
    gs->granularity[1] = granularity[1];

    // 1) The tiling continues until the entire output matrix of size I,J has been tiled.
    // 2) global_sizes must be near to opimal_tiling_size
    // 3) global sizes must be multiple of local_sizes. That's why we call it granularity.
    // 4) local_sizes must be near, but less or equal to work_group_multiplier_hint (thats is a warp size)

    // scale to granularity
    {
        // some complicated tiling example (with granularity:1 or downscale with granularity)
        // size: 6, optimal_tile_size: 5; then 5,1 is a very bad tiling. 3,3 is mutch better.
        // size: 16, optimal_tile_size: 5; then 5,5,5,1 is a bad tiling. 4,4,4,4 is mutch better
        // size: 18, optimal_tile_size: 5; then 5,5,5,3 is a bad tiling. 5,5,4,4 is mutch better
        // for size:21 the optimal tiling is 5,4,4,4,4.

        //printf("    DEBUG optimal_tiling_size:%f\n", gs->opimal_tiling_size);
        size_t optimal_tilesize0 = ((int)floor(gs->opimal_tiling_size) / granularity[0]);  // round down: not exceed the cache size
        size_t optimal_tilesize1 = ((int)floor(gs->opimal_tiling_size) / granularity[1]);  // round down: not exceed the cache size
        if( optimal_tilesize0 <1 ){
            dprintf(2, "Warning: matrix size is too large for fit any tile in the cache. Continue anyway.\n");
            optimal_tilesize0 =  1;
        }
        if( optimal_tilesize1 <1 ){
            dprintf(2, "Warning: matrix size is too large for fit any tile in the cache. Continue anyway.\n");
            optimal_tilesize1 =  1;
        }
        //printf("    DEBUG optimal_tilesize:(%lu, %lu)\n", optimal_tilesize0, optimal_tilesize1);

        size_t matrixsize0 = divideup(matrix_size[0], granularity[0]);               // round up: the matrix must be fit
        size_t matrixsize1 = divideup(matrix_size[1], granularity[1]);               // round up: the matrix must be fit
        //printf("    DEBUG matrixsize:(%lu, %lu)\n", matrixsize0, matrixsize1);
        // how many tiles?
        gs->tilenumber[0] = divideup(matrixsize0, optimal_tilesize0);
        gs->tilenumber[1] = divideup(matrixsize1, optimal_tilesize1);
        //printf("    DEBUG tilenumber:(%lu, %lu)\n",  gs->tilenumber[0],  gs->tilenumber[1]);
        gs->base_tilesize[0] = matrixsize0 / gs->tilenumber[0];
        gs->base_tilesize[1] = matrixsize1 / gs->tilenumber[1];
        //printf("    DEBUG base_tilesize:(%lu, %lu)\n", gs->base_tilesize[0], gs->base_tilesize[1]);
        gs->tilesize_remainder[0] = matrixsize0 % gs->tilenumber[0];
        gs->tilesize_remainder[1] = matrixsize1 % gs->tilenumber[1];
        //printf("    DEBUG tilesize_remainder:(%lu, %lu)\n", gs->tilesize_remainder[0], gs->tilesize_remainder[1]);
        gs->current_tilenum[0] = 0;
        gs->current_tilenum[1] = 0;
        gs->row_increment = 1;
    }

    return 1;
}



int gener_tiling_next(
    struct generstate * gs,
    size_t * offset,   /* 2-len vector output offset (which tile?) rounded to granularity*/
    size_t * tilesize       /*  2-len vector output tile size. (rounded to granularity) eg opencl's global_size*/
    )
{
    if( NULL == gs){
        dprintf(2, "Err: NULL generstate in gener_tiling\n");
        exit(2);
    }
    if( GENERSTATE_MAGIC != gs->magic){
        dprintf(2, "Err: wrong magic in generstate at gener_tiling\n");
        exit(2);
    }

    //if end of the iter?
    if(gs->current_tilenum[0] == gs->tilenumber[0]){
        offset[0] = offset[1] = 0;
        tilesize[0] = tilesize[1] = 0;
        gs->magic = 0;
        return 0; // end iteration
    }
    //printf("   D current_tilenum:(%lu, %lu) base_tilesize:(%lu, %lu)\n",
    //        gs->current_tilenum[0], gs->current_tilenum[1], gs->base_tilesize[0], gs->base_tilesize[1]);

    // get the current tile
    if( gs->current_tilenum[0] < gs->tilesize_remainder[0]){
        offset[0] =  gs->current_tilenum[0] * (gs->base_tilesize[0] + 1) * gs->granularity[0];
        tilesize[0] = (gs->base_tilesize[0] + 1) * gs->granularity[0];
    } else {
        offset[0] =  (gs->current_tilenum[0] * gs->base_tilesize[0] + gs->tilesize_remainder[0]) * gs->granularity[0];
        tilesize[0] = gs->base_tilesize[0] * gs->granularity[0];
    }
    if( gs->current_tilenum[1] < gs->tilesize_remainder[1]){
        offset[1] =  gs->current_tilenum[1] * (gs->base_tilesize[1] + 1) * gs->granularity[1];
        tilesize[1] = (gs->base_tilesize[1] + 1 ) * gs->granularity[1];
    } else {
        offset[1] =  (gs->current_tilenum[1] * gs->base_tilesize[1] + gs->tilesize_remainder[1]) * gs->granularity[1];
        tilesize[1] = gs->base_tilesize[1] * gs->granularity[1];
    }


    // advance the current tilenum in the row
    // end of the row? then advance the row itself
    if( -1 == gs->row_increment && 0 == gs->current_tilenum[1]){
        gs->row_increment = 1;
        gs->current_tilenum[0]++;
    } else if( 1 == gs->row_increment &&  gs->current_tilenum[1] + 1 ==  gs->tilenumber[1]){
        gs->row_increment = -1;
        gs->current_tilenum[0]++;
    } else {
        gs->current_tilenum[1] += gs->row_increment;
    }
    return 1;
}/* end of gener_tiling_next */
