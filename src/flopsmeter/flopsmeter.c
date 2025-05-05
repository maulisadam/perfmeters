/* flopsmeter.c
**
** measure Flops via the matrix multiplication
**
** purely for perfomance measuerement purposes
**
**
**  Copyright by Adam Maulis, 2025  In terms of GNU GPL v3 or newer
**
**
**  gcc -O4 -o flopsmeter  flopsmeter.c -march=native -DUSE_RESTRICT -Wall  -ffast-math -lpthread
*/


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pthread.h>

#define PRECISION 80
#include "matmul.inc"
#undef PRECISION
#define PRECISION 64
#include "matmul.inc"
#undef PRECISION
#define PRECISION 32
#include "matmul.inc"
#undef PRECISION


/* 
** time differencial utility function: returns int (nanoseconds)
*/
static inline long int diff_timespec(const struct timespec *endtime, const struct timespec *begtime) {
  return (endtime->tv_sec - begtime->tv_sec) * 1000000000L +
            (endtime->tv_nsec - begtime->tv_nsec);
}


int main(int argc, const char * argv[])
{
    struct timespec begintime, inittime, endtime;
    double flop;


    if( argc != 4){
        dprintf(2,"Usage: ./flopsmeter n_X n_Y n_Z\n");
        dprintf(2,"\tMeasure the Flops performance via matrix multiplication: R = M1 x M2");
        dprintf(2,"\tn_X n_Y n_Z are integers, the shape of the matricies:\n");
        dprintf(2,"\tM1 is (n_X, n_Y), M2 is (n_Y, n_Z), R is (n_X, n_Z)\n");
        return 2;
    }

    /* calculate flops*/
    flop = atof(argv[2]) * atof(argv[1]) * atof(argv[3]);      /* multiplications */
    flop += (atof(argv[2]) -1) *  atof(argv[1]) * atof(argv[3]); /* additions */

    /*
    dmatrix_load(&din1, argv[1]);
    dmatrix_load(&din2, argv[2]);
    dmatrix_init(&din2_t, din2.j, din2.i);
    dmatrix_transpose(&din2, &din2_t);
    */
    {
        ldmatrix in1, in2, out;
        clock_gettime(CLOCK_REALTIME, &begintime);
        ldmatrix_init(&in1, atoi(argv[1]), atoi(argv[2]));
        ldmatrix_init(&in2, atoi(argv[3]), atoi(argv[2])); /* random initialization as transposed */
        ldmatrix_init(&out, in1.i, in2.i );
        ldmatrix_random(&in1);
        ldmatrix_random(&in2);
        clock_gettime(CLOCK_REALTIME, &inittime);
        ldmatmul_t(&in1, &in2, &out);
        clock_gettime(CLOCK_REALTIME, &endtime);
        ldmatrix_free(&in1);
        ldmatrix_free(&in2);
        ldmatrix_free(&out);
        printf("{\"n_X\":%d, \"n_Y\":%d, \"n_Z\":%d, \"dtype\":\"%s\", \"time_init\":%f, \"time_matmul\":%f, \"Gflops\":%f}\n",
            atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
            "fp80",
            diff_timespec(&inittime, &begintime)/1000000000.0,
            diff_timespec(&endtime, &inittime)/1000000000.0,
            flop/diff_timespec(&endtime, &inittime)
        );
    }
    {
        dmatrix in1, in2, out;
        clock_gettime(CLOCK_REALTIME, &begintime);
        dmatrix_init(&in1, atoi(argv[1]), atoi(argv[2]));
        dmatrix_init(&in2, atoi(argv[3]), atoi(argv[2])); /* random initialization as transposed */
        dmatrix_init(&out, in1.i, in2.i );
        dmatrix_random(&in1);
        dmatrix_random(&in2);
        clock_gettime(CLOCK_REALTIME, &inittime);
        dmatmul_t(&in1, &in2, &out);
        clock_gettime(CLOCK_REALTIME, &endtime);
        dmatrix_free(&in1);
        dmatrix_free(&in2);
        dmatrix_free(&out);
        printf("{\"n_X\":%d, \"n_Y\":%d, \"n_Z\":%d, \"dtype\":\"%s\", \"time_init\":%f, \"time_matmul\":%f, \"Gflops\":%f}\n",
            atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
            "fp64",
            diff_timespec(&inittime, &begintime)/1000000000.0,
            diff_timespec(&endtime, &inittime)/1000000000.0,
            flop/diff_timespec(&endtime, &inittime)
        );
    }
    {
        fmatrix in1, in2, out;
        clock_gettime(CLOCK_REALTIME, &begintime);
        fmatrix_init(&in1, atoi(argv[1]), atoi(argv[2]));
        fmatrix_init(&in2, atoi(argv[3]), atoi(argv[2])); /* random initialization as transposed */
        fmatrix_init(&out, in1.i, in2.i );
        fmatrix_random(&in1);
        fmatrix_random(&in2);
        clock_gettime(CLOCK_REALTIME, &inittime);
        fmatmul_t(&in1, &in2, &out);
        clock_gettime(CLOCK_REALTIME, &endtime);
        fmatrix_free(&in1);
        fmatrix_free(&in2);
        fmatrix_free(&out);
        printf("{\"n_X\":%d, \"n_Y\":%d, \"n_Z\":%d, \"dtype\":\"%s\", \"time_init\":%f, \"time_matmul\":%f, \"Gflops\":%f}\n",
            atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
            "fp32",
            diff_timespec(&inittime, &begintime)/1000000000.0,
            diff_timespec(&endtime, &inittime)/1000000000.0,
            flop/diff_timespec(&endtime, &inittime)
        );
    }

    /*
    dmatrix_save(&out2, argv[3]);
    clock_gettime(CLOCK_REALTIME, &endtime);
    */
}/* end of main */
