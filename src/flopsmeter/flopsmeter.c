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
#include <string.h>
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
** grep -m 'model name\t: ' /proc/cpuinfo
**   non posix thread reentrant!
**  good old K&R: This code is so crappy, don't use it as an example.
*/
static char * get_cpu_model_name(void)
{
    static char buff[3000]; /* longest /proc/cpuinfo line legth is 800 at this time */
    static int cache = 0;
    FILE * fd;
    size_t linelen;
    int i;
    char * matchstring = "model name\t: ";
    int matchstringlen = strlen(matchstring);

    //printf("D matchstringlen=%d\n", matchstringlen);
    if( 0 != cache){
        return buff;
    }

    fd = fopen("/proc/cpuinfo", "r");
    if( NULL == fd){
        dprintf(2, "Warning: error in open /proc/cpuinfo\n");
        return "unknown cpu";
    }
    buff[0] = '\n';
    while( 0 == cache && 0 != buff[0]){
        fgets(buff, 3000, fd);
        linelen = strlen(buff);
        //printf("D loaded line: »%*s«\n", matchstringlen+1, buff);
        if( '\n' != buff[linelen-1] ){
            fclose(fd);
            dprintf(2, "Warning: error in reading line from /proc/cpuinfo\n");
            return "unknown cpu";
        }
        if( 0 == strncmp(buff, matchstring, matchstringlen)){
            /* overlap strcpy */
            for(i=0; i < linelen - matchstringlen; i++){
                if( '\n' == buff[ i+matchstringlen]){
                    buff[i]=0;
                }else{
                    buff[i] = buff[ i+matchstringlen];
                }
            }
            cache = 1;
        }
    }
    fclose(fd);
    if(0 == cache){
        return "unknown cpu";
    }else{
        return buff;
    }
}


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
    int threadcnt;


    if( argc != 4){
        dprintf(2,"Usage: ./flopsmeter n_X n_Y n_Z\n");
        dprintf(2,"\tMeasure the Flops performance via matrix multiplication: R = M1 x M2");
        dprintf(2,"\tn_X n_Y n_Z are integers, the shape of the matricies:\n");
        dprintf(2,"\tM1 is (n_X, n_Y), M2 is (n_Y, n_Z), R is (n_X, n_Z)\n");
        return 2;
    }

    // get cpu count
    threadcnt = sysconf(_SC_NPROCESSORS_ONLN);
    if(threadcnt < 1){
        dprintf(2, "Err: main invalid cpu count. Cannot create threads\n");
        exit(1);
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
        printf("{\"n_X\":%d, \"n_Y\":%d, \"n_Z\":%d, \"dtype\":\"%s\", \"threadcount\":%d, \"device_type\":\"%s\", \"device\":\"%s\", \"time_init\":%f, \"time_matmul\":%f, \"Gflops\":%f}\n",
            atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
            "fp80",
            threadcnt,
            "pthread-cpu",
            get_cpu_model_name(),
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
        printf("{\"n_X\":%d, \"n_Y\":%d, \"n_Z\":%d, \"dtype\":\"%s\", \"threadcount\":%d, \"device_type\":\"%s\", \"device\":\"%s\", \"time_init\":%f, \"time_matmul\":%f, \"Gflops\":%f}\n",
            atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
            "fp64",
            threadcnt,
            "pthread-cpu",
            get_cpu_model_name(),
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
        printf("{\"n_X\":%d, \"n_Y\":%d, \"n_Z\":%d, \"dtype\":\"%s\", \"threadcount\":%d, \"device_type\":\"%s\", \"device\":\"%s\", \"time_init\":%f, \"time_matmul\":%f, \"Gflops\":%f}\n",
            atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
            "fp32",
            threadcnt,
            "pthread-cpu",
            get_cpu_model_name(),
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
