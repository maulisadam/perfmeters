/* matmul_test.c
**
** matmul implementatiosn testing
**
** purely for perfomance measuerement purposes
**
**
**  Copyright by Adam Maulis, 2025  In terms of GNU AGPL v3 or newer
**
*/


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "matmul.h"

/*
** time differencial utility function: returns int (nanoseconds)
*/
static inline long int diff_timespec(const struct timespec *endtime, const struct timespec *begtime) {
  return (endtime->tv_sec - begtime->tv_sec) * 1000000000L +
            (endtime->tv_nsec - begtime->tv_nsec);
}


int main(int argc, const char * argv[])
{
    struct timespec begintime, endtime;
    double flop;

    if( argc != 5){
        dprintf(2,"Usage: ./matmul_test PREC A B C\n");
        dprintf(2,"\tDo the matrix multiplication where C = A x B");
        dprintf(2,"\tPREC is one of the following: fp16, fp32, fp64, fp80, fp128\n");
        dprintf(2,"\tA and B is existing files on format of numpy.savetxt()\n");
        dprintf(2,"\tC will be same format.\n");

        return 2;
    }

    if( 0 == strcmp(argv[1], "fp16")){
#ifdef FLT16_MIN
        hmatrix in1, in2, in2_t, out;

        hmatrix_load(&in1, argv[2]);
        hmatrix_load(&in2, argv[3]);
        hmatrix_init(&in2_t, in2.j, in2.i);
        hmatrix_transpose(&in2, &in2_t);
        /* calculate flops*/
        flop = (double)in1.j * (double)in1.i * (double)in2.j  ;      /* multiplications */
        flop += ((double)in1.j -1) *  (double)in1.i * (double)in2.j; /* additions */

        hmatrix_free(&in2);
        hmatrix_init(&out, in1.i, in2_t.i);
        clock_gettime(CLOCK_REALTIME, &begintime);
        hmatmul_t(&in1, &in2_t, &out(;
        clock_gettime(CLOCK_REALTIME, &endtime);
        hmatrix_save(&out, argv[4]);
        hmatrix_free(&in1);
        hmatrix_free(&in2_t);
        hmatrix_free(&out);

        printf("Precision: %s, time_matmul:%f Gflops:%f\n",
            argv[1],
            diff_timespec(&endtime, &begintime)/1000000000.0,
            flop/diff_timespec(&endtime, &begintime)
            );


#else
        dprintf(2, "Warn: fp16 is not supported this platform\n");
#endif

    } else if( 0 == strcmp(argv[1], "fp32")){
        fmatrix in1, in2, in2_t, out;

        fmatrix_load(&in1, argv[2]);
        fmatrix_load(&in2, argv[3]);
        fmatrix_init(&in2_t, in2.j, in2.i);
        fmatrix_transpose(&in2, &in2_t);
        /* calculate flops*/
        flop = (double)in1.j * (double)in1.i * (double)in2.j  ;      /* multiplications */
        flop += ((double)in1.j -1) *  (double)in1.i * (double)in2.j; /* additions */

        fmatrix_free(&in2);
        fmatrix_init(&out, in1.i, in2_t.i);
        clock_gettime(CLOCK_REALTIME, &begintime);
        fmatmul_t(&in1, &in2_t, &out);
        clock_gettime(CLOCK_REALTIME, &endtime);
        fmatrix_save(&out, argv[4]);
        fmatrix_free(&in1);
        fmatrix_free(&in2_t);
        fmatrix_free(&out);

        printf("Precision: %s, time_matmul:%f Gflops:%f\n",
            argv[1],
            diff_timespec(&endtime, &begintime)/1000000000.0,
            flop/diff_timespec(&endtime, &begintime)
            );

    } else if( 0 == strcmp(argv[1], "fp64")){
        dmatrix in1, in2, in2_t, out;

        dmatrix_load(&in1, argv[2]);
        dmatrix_load(&in2, argv[3]);
        dmatrix_init(&in2_t, in2.j, in2.i);
        dmatrix_transpose(&in2, &in2_t);
        /* calculate flops*/
        flop = (double)in1.j * (double)in1.i * (double)in2.j  ;      /* multiplications */
        flop += ((double)in1.j -1) *  (double)in1.i * (double)in2.j; /* additions */

        dmatrix_free(&in2);
        dmatrix_init(&out, in1.i, in2_t.i);
        clock_gettime(CLOCK_REALTIME, &begintime);
        dmatmul_t(&in1, &in2_t, &out);
        clock_gettime(CLOCK_REALTIME, &endtime);
        dmatrix_save(&out, argv[4]);
        dmatrix_free(&in1);
        dmatrix_free(&in2_t);
        dmatrix_free(&out);

        printf("Precision: %s, time_matmul:%f Gflops:%f\n",
            argv[1],
            diff_timespec(&endtime, &begintime)/1000000000.0,
            flop/diff_timespec(&endtime, &begintime)
            );

    } else if( 0 == strcmp(argv[1], "fp80")){
        ldmatrix in1, in2, in2_t, out;

        ldmatrix_load(&in1, argv[2]);
        ldmatrix_load(&in2, argv[3]);
        ldmatrix_init(&in2_t, in2.j, in2.i);
        ldmatrix_transpose(&in2, &in2_t);
        /* calculate flops*/
        flop = (double)in1.j * (double)in1.i * (double)in2.j  ;      /* multiplications */
        flop += ((double)in1.j -1) *  (double)in1.i * (double)in2.j; /* additions */

        ldmatrix_free(&in2);
        ldmatrix_init(&out, in1.i, in2_t.i);
        clock_gettime(CLOCK_REALTIME, &begintime);
        ldmatmul_t(&in1, &in2_t, &out);
        clock_gettime(CLOCK_REALTIME, &endtime);
        ldmatrix_save(&out, argv[4]);
        ldmatrix_free(&in1);
        ldmatrix_free(&in2_t);
        ldmatrix_free(&out);

        printf("Precision: %s, time_matmul:%f Gflops:%f\n",
            argv[1],
            diff_timespec(&endtime, &begintime)/1000000000.0,
            flop/diff_timespec(&endtime, &begintime)
            );

    } else if( 0 == strcmp(argv[1], "fp128")){
        qmatrix in1, in2, in2_t, out;

        qmatrix_load(&in1, argv[2]);
        qmatrix_load(&in2, argv[3]);
        qmatrix_init(&in2_t, in2.j, in2.i);
        qmatrix_transpose(&in2, &in2_t);
        /* calculate flops*/
        flop = (double)in1.j * (double)in1.i * (double)in2.j  ;      /* multiplications */
        flop += ((double)in1.j -1) *  (double)in1.i * (double)in2.j; /* additions */

        qmatrix_free(&in2);
        qmatrix_init(&out, in1.i, in2_t.i);
        clock_gettime(CLOCK_REALTIME, &begintime);
        qmatmul_t(&in1, &in2_t, &out);
        clock_gettime(CLOCK_REALTIME, &endtime);
        qmatrix_save(&out, argv[4]);
        qmatrix_free(&in1);
        qmatrix_free(&in2_t);
        qmatrix_free(&out);

        printf("Precision: %s, time_matmul:%f Gflops:%f\n",
            argv[1],
            diff_timespec(&endtime, &begintime)/1000000000.0,
            flop/diff_timespec(&endtime, &begintime)
            );

    } else {
        dprintf(2,"Err: unknown PREC. Must be one of the following: fp16, fp32, fp64, fp80, fp128\n");
    }
}/* end of main */
