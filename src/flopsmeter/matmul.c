/* matmul.c
**
** implementation of the matrix multiplication
**
** purely for perfomance measuerement purposes
**
**
**  Copyright by Adam Maulis, 2025
**
**  In terms of GNU GPL v3 or newer
**
**
**  gcc -O4 -o matmul matmul.c -march=native -DUSE_RESTRICT -Wall -ffast-math
**  gcc -O4 -o matmul matmul.c -march=native -DUSE_RESTRICT -Wall  -ffast-math -lpthread
*/


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pthread.h>


#ifdef USE_RESTRICT
#define _RESTRICT restrict
#else
#define _RESTRICT
#endif


/*
** time differencial utility function: returns int (nanoseconds)
*/
static inline long int diff_timespec(const struct timespec *endtime, const struct timespec *begtime) {
  return (endtime->tv_sec - begtime->tv_sec) * 1000000000L +
            (endtime->tv_nsec - begtime->tv_nsec);
}

#define DMATRIX_MAGIC 4294967087u

typedef struct {
    double * * _RESTRICT m;
    uint32_t i, j;
    uint32_t magic;
} dmatrix;

static inline void dmatrix_assert(const dmatrix * _RESTRICT dp, const char * msg, const char * file, const int line)
{
    if( NULL == dp){
        dprintf(2,"Err in %s: not a dmatrix %s:%d\n", msg, file, line);
        exit(2);
    }
    if( NULL == dp->m ){
        dprintf(2,"Err in %s : null dmatrix %s:%d\n", msg, file, line);
        exit(2);
    }
    if( NULL == dp->m[0] ){
        dprintf(2,"Err in %s : null dmatrix row %s:%d\n", msg, file, line);
        exit(2);
    }
    if( DMATRIX_MAGIC != dp->magic ){
        dprintf(2,"Err in %s : wrong magic dmatrix %s:%d\n", msg, file, line);
        exit(2);
    }
    if( 0 == dp->i || 0 == dp->j ){
        dprintf(2,"Err in %s : null size dmatrix %s:%d\n", msg, file, line);
        exit(2);
    }

}

static inline void dmatrix_init_assert(const dmatrix * _RESTRICT dp, const char * msg, const char * file, const int line)
{
    if( NULL == dp){
        dprintf(2,"Err in init: not a dmatrix %s:%d\n", file, line);
        exit(2);
    }
    if( DMATRIX_MAGIC == dp->magic){
        dprintf(2,"Err in init: used dmatrix, no reinit %s:%d\n", file, line);
        exit(2);
    }
}

#ifndef ROWALIGN
#define ROWALIGN 16
#endif

void _dmatrix_init(dmatrix * dp, uint32_t i, uint32_t j, const char* file, const int line)
{
    uint32_t ii;
    uint32_t roundup_j;

    dmatrix_init_assert(dp, "init", file, line);

    if( 0==i || 0 == j){
        dprintf(2,"Err in init: invalid dmatrix size(%u, %u) %s:%d\n", i,j, file, line);
        exit(2);
    }
    dp->m = (double **) malloc( sizeof(double *) * i);
    if( NULL == dp->m ){
        dprintf(2,"Err in init: ENOMEM for malloc %s:%d\n", file, line);
        exit(2);
    }

    roundup_j = j;
    if( 0 != (j % ROWALIGN)){
        roundup_j = (j/ROWALIGN +1) * ROWALIGN;
    }

    for(ii=0; ii<i; ii++){
        dp->m[ii] = (double *) aligned_alloc(ROWALIGN, sizeof(double) * roundup_j);
        if( NULL == dp->m[ii] ){
            dprintf(2,"Err in init: ENOMEM for row malloc %s:%d\n", file, line);
            exit(2);
        }
    }
    dp->i = i;
    dp->j = j;
    dp->magic = DMATRIX_MAGIC;
}
#define dmatrix_init(a,b,c)  _dmatrix_init(a,b,c, __FILE__, __LINE__)


/*
** Load file saved as python's numpy module:
**
** >>> np.savetxt("B.txt", B)
*/
void _dmatrix_load(dmatrix * dp, const char * filename, const char* file, const int line)
{

    int fd, retval, height, width, i, j, old_ii, ii;
    char * buff;
    struct stat statbuf;

    //puts("D load beg");
    dmatrix_init_assert(dp, "load", file, line);
    fd = open(filename, O_RDONLY);
    if( fd<0 ){
        perror("Error while opening file");
        exit(1);
    }
    retval = fstat(fd, &statbuf);
    if( retval <0 ){
        perror("Error while inquire file size");
        exit(1);
    }
    //puts("D load before malloc");

    buff = (char*) malloc(statbuf.st_size);
    if(NULL == buff){
        dprintf(2, "Error in load: no enought memory. %s:%d\n", file, line);
        exit(2);
    }
    //puts("D load after malloc");
    retval = read(fd, buff, statbuf.st_size);
    if( retval <0 ){
        perror("Error while file read");
        exit(1);
    }
    //puts("D load after read");
    close(fd);

    /* determine height(i) and width(j) */
    height = 0;
    for(i=0; i<statbuf.st_size; ++i){
        if('\n' == buff[i]){
            height++;
        }
    }
    width = 1;
    for(ii=0; ii<statbuf.st_size; ++ii){
        if('\n' == buff[ii]){
            break;
        }
        if(' ' == buff[ii]){
            width++;
        }
    }
    //printf("D load height:%u, width:%u\n", height, width);
    _dmatrix_init(dp, height, width, file, line);
    //puts("D load after init");

    /* convert in-memory text to dmatrix */
    old_ii = 0;
    i = j = 0;
    for(ii=0; ii<statbuf.st_size; ++ii){
        if((buff[ii] != '\n') & (buff[ii] != ' ')){
            continue;
        }
        //printf("D load main convert loop. m i:%d j:%d offset:%d, old_ii:%d, ii:%d ", i, j, i*dp->j + j, old_ii, ii);
        //printf("value:%f\n", atof(buff+old_ii));
        dp->m[i][j] = atof(buff+old_ii);
        if( j+1 == dp->j){
            i = i+1;
            j = 0;
        } else {
            j = j+1;
        }
        old_ii = ii+1;
    }/* end for ii */

    free(buff);
}
#define dmatrix_load(a,b) _dmatrix_load(a,b, __FILE__, __LINE__)


void _dmatrix_free(dmatrix * dp, const char* file, const int line)
{
    uint32_t ii;

    dmatrix_assert(dp, "free", file, line);
    for(ii=0; ii<dp->i; ii++){
        free(dp->m[ii]);
        dp->m[ii]=NULL;
    }
    free(dp->m);
    dp->m = NULL;
    dp->i = 0;
    dp->j = 0;
    dp->magic = 0;
}
#define dmatrix_free(a)  _dmatrix_init(a, __FILE__, __LINE__)


/* fill a random */
void _dmatrix_random(dmatrix * dp,  const char * file, const int line)
{
    uint32_t ii, jj;

    dmatrix_assert(dp, "random", file, line);

    for(ii=0; ii < dp->i; ii++){
        for(jj=0; jj < dp->j; ++jj){
            //dp->m[ii][jj] = drand48();
            dp->m[ii][jj] = ii+jj+1;
        }
    }

}
#define dmatrix_random(a)  _dmatrix_random(a, __FILE__, __LINE__)


/* transpose */
void _dmatrix_transpose(const dmatrix * in, dmatrix * out,  const char * file, const int line)
{
    uint32_t ii, jj;

    dmatrix_assert(in, "transpose", file, line);
    dmatrix_assert(out, "transpose", file, line);
    if( in->i != out->j || in->j != out->i){
        dprintf(2, "Err in transpose: in (%u,%u) and out(%u,%u) dmatrix shape mismatch %s:%d\n",
            in->i, in->j, out->i, out->j, file, line);
        exit(2);
    }

    for(ii=0; ii < in->i; ++ii){
        for(jj=0; jj < in->j; ++jj){
            out->m[jj][ii] = in->m[ii][jj];
        }
    }
}
#define dmatrix_transpose(a,b)  _dmatrix_transpose(a,b, __FILE__, __LINE__)


void _dmatrix_print(const dmatrix * dp, const char * file, const int line)
{
    uint32_t i,j;

    dmatrix_assert(dp, "print", file, line);
    for(i=0; i< dp->i; ++i){
        for(j=0; j< dp->j; ++j){
            printf("%.5f ", dp->m[i][j]);
        }
        puts("");
    }
}
#define dmatrix_print(a) _dmatrix_print(a, __FILE__, __LINE__)

void _dmatrix_save(const dmatrix * dp, const char * savefile, const char * file, const int line)
{
    uint32_t i,j;
    FILE * sfd;

    dmatrix_assert(dp, "save", file, line);

    sfd = fopen(savefile, "w");
    if( NULL == sfd){
        perror("Error in output save");
        exit(1);
    }
    for(i=0; i< dp->i; ++i){
        for(j=0; j< dp->j-1; ++j){
            fprintf(sfd, "%.18e ", dp->m[i][j]);
        }
        fprintf(sfd, "%.18e\n", dp->m[i][j]);
    }
    fclose(sfd);
}
#define dmatrix_save(a, b) _dmatrix_save(a,b, __FILE__, __LINE__)


/*******************************************************************/

/*schoolbook implementation */
void _dmatmul(const dmatrix * _RESTRICT in1, const dmatrix * _RESTRICT in2, dmatrix * _RESTRICT out, const char * file, const int line)
{
    uint32_t oi, oj, ii;

    dmatrix_assert(in1, "dmatmul:in1", file, line);
    dmatrix_assert(in2, "dmatmul:in2", file, line);
    dmatrix_assert(out, "dmatmul:out", file, line);


    if(in1->i != out->i){
        dprintf(2,"Err: dmatmul: in1 versus out shape mismatch\n");
        exit(2);
    }
    if(in2->j != out->j){
        dprintf(2,"Err: dmatmul: in2 versus out shape mismatch\n");
        exit(2);
    }
    if(in1->j != in2->i){
        dprintf(2,"Err: dmatmul: in1 versus in2 shape mismatch\n");
        exit(2);
    }

    for(oi=0; oi < out->i; ++oi){
        for(oj=0; oj < out->j; ++oj){
            out->m[oi][oj]  = 0.0;
            for(ii=0; ii<in2->i; ++ii){
                 out->m[oi][oj] += in1->m[oi][ii] * in2->m[ii][oj];
            }
        }
    }
}
#define dmatmul(a,b,c) _dmatmul(a,b,c, __FILE__, __LINE__)

static inline double dvector_dot(const double * _RESTRICT v1, const double * _RESTRICT v2, const uint32_t len)
{
    double register retval;
    uint32_t i;

    retval = *v1 * *v2;
    for(i=1; i < len; i++){
        retval += v1[i]*v2[i];
    }
    /*
    for(i=0; i < len; i++){
        tmp[i] = v1[i]*v2[i];
    }
    retval = tmp[0]; // this 'tmp' makes things slower.
    for(i=1; i < len; i++){
        retval += tmp[i];
    }
    */
    return retval;
}

/*
    optimalized one. requires a second matrix as transposed
    multithreaded
*/

struct dmatmul_t_threadargs{
    dmatrix * _RESTRICT in1;
    dmatrix * _RESTRICT in2;
    dmatrix * _RESTRICT out;
    uint32_t from;
    uint32_t to;
};


static inline void _dmatmul_t_sub(
        const dmatrix * _RESTRICT in1,
        const dmatrix * _RESTRICT in2,
        dmatrix * _RESTRICT out,
        const uint32_t from,
        const uint32_t to)
{
    uint32_t oi, oj;
    double * _RESTRICT rowpointer1;
    double * _RESTRICT rowpointer2;

    for(oi=from; oi < to; oi++){
        rowpointer1 = in1->m[oi];
        for(oj=0; oj<out->j; oj++){
            rowpointer2 = in2->m[oj];
            out->m[oi][oj] = dvector_dot(rowpointer1, rowpointer2, in1->j);
        }
    }
}


static void * _dmatmul_t_thread(void *arg)
{
    struct dmatmul_t_threadargs * myarg = (struct dmatmul_t_threadargs *) arg;
    _dmatmul_t_sub( myarg->in1, myarg->in2, myarg->out, myarg->from, myarg->to);
    return NULL;
}


void _dmatmul_t(dmatrix * _RESTRICT in1, dmatrix * _RESTRICT in2, dmatrix * _RESTRICT out, const char * file, const int line)
{

    dmatrix_assert(in1, "dmatmul_t:in1", file, line);
    dmatrix_assert(in2, "dmatmul_t:in2", file, line);
    dmatrix_assert(out, "dmatmul_t:out", file, line);

    int threadcnt, i;
    pthread_t * threadp;
    struct dmatmul_t_threadargs * threadargs;

    if(in1->i != out->i){
        dprintf(2,"Err: dmatmul_t: in1 versus out shape mismatch %s:%i\n", file, line);
        exit(2);
    }
    if(in2->i != out->j){
        dprintf(2,"Err: dmatmul_t: in2 versus out shape mismatch %s:%i\n", file, line);
        exit(2);
    }
    if(in1->j != in2->j){
        dprintf(2,"Err: dmatmul_t: in1 versus in2 shape mismatch %s:%i\n", file, line);
        exit(2);
    }
    // get cpu count
    threadcnt = sysconf(_SC_NPROCESSORS_ONLN);
    if(threadcnt < 1){
        dprintf(2, "Err: dmatmul_t: invalid cpu count. Cannot create threads\n");
        exit(1);
    }
    //printf("D threadcnt:%d\n", threadcnt);
    threadp = (pthread_t * ) malloc( threadcnt * sizeof(pthread_t));
    threadargs = (struct dmatmul_t_threadargs *) malloc( threadcnt * sizeof(struct dmatmul_t_threadargs));
    if(NULL == threadp || NULL == threadargs){
        dprintf(2, "Err: dmatmul_t: cannot allocate memory for threads\n");
        exit(1);
    }
    // pthread_create
    for ( i=0; i < threadcnt; ++i ) {
        int retval;
        threadargs[i].in1 = in1;
        threadargs[i].in2 = in2;
        threadargs[i].out = out;
        threadargs[i].from = i * ((out->i) / (uint32_t)threadcnt);
        if( i == threadcnt-1){
            threadargs[i].to = out->i;
        }else{
            threadargs[i].to = (i+1) * ((out->i) / (uint32_t)threadcnt);
        }
        retval = pthread_create(&(threadp[i]), 0, &_dmatmul_t_thread, threadargs+i);
        if( retval ) {
            dprintf(2, "Err: dmatmul_t thread creation. Error code: %d\n", retval);
            exit(1);
        }
    }

    // pthread_join
    for ( i=0; i < threadcnt; ++i ) {
        int retval;

        retval = pthread_join(threadp[i], 0);
        if( retval ) {
            dprintf(2, "Err: dmatmul_t: cannot join worker threads (%d)\n", i);
            exit(1);
        }
    }


}
#define dmatmul_t(a,b,c) _dmatmul_t(a,b,c, __FILE__, __LINE__)


int main(int argc, const char * argv[])
{
    struct timespec t1, t2, t3, t4, endtime;
    long int difftime;

    dmatrix in1, in2, in2_t, out1, out2;

    if( argc != 4){
        dprintf(2,"Usage: ./matmul infile1 infile2 outf\n");
        dprintf(2,"  infile1: first matrix in numpy savetxt format\n");
        dprintf(2,"  infile2: second matrix in numpy savetxt format\n");
        dprintf(2,"  outf: result file name\n");
        return 2;
    }
    clock_gettime(CLOCK_REALTIME, &t1);
    //puts("D 1");
    dmatrix_load(&in1, argv[1]);
    //puts("D 2");
    dmatrix_load(&in2, argv[2]);
    dmatrix_init(&in2_t, in2.j, in2.i);
    dmatrix_transpose(&in2, &in2_t);
    //puts("D 3");
    dmatrix_init(&out1, in1.i, in2.j );
    //puts("D 4");
    dmatrix_init(&out2, in1.i, in2.j );
    //dmatrix_random(&in1);
    //dmatrix_random(&in2);
    clock_gettime(CLOCK_REALTIME, &t2);

    //dmatmul(&in1, &in2, &out1);
    clock_gettime(CLOCK_REALTIME, &t3);

    dmatmul_t(&in1, &in2_t, &out2);
    clock_gettime(CLOCK_REALTIME, &t4);
    dmatrix_save(&out2, argv[3]);
    clock_gettime(CLOCK_REALTIME, &endtime);

    /* calculate flops*/
    double flop = (double)in1.j * (double)out2.i * (double)out2.j;      /* multiplications */
    //double flop1 = flop*2; /* schoolbook implementation has an additional addition */
    flop += (double)(in1.j-1) * (double)out2.i * (double)out2.j; /* additions */


    difftime = diff_timespec(&t2, &t1);
    printf("Initialization time: %f\n", difftime/1000000000.0);
    //difftime = diff_timespec(&t3, &t2);
    //printf("Matmul (schoolbook) time: %f Gflops: %f\n", difftime/1000000000.0, flop1/(difftime));
    difftime = diff_timespec(&t4, &t3);
    printf("Matmul_t time: %f Gflops: %f (flopcount: %f)\n", difftime/1000000000.0, flop/(difftime), flop);

    difftime = diff_timespec(&endtime, &t4);
    printf("Save time: %f\n", difftime/1000000000.0);

    if(0){
        puts("in1 -----");
        dmatrix_print(&in1);
        puts("in2 -----");
        dmatrix_print(&in2);
        puts("in2_t -----");
        dmatrix_print(&in2_t);
        puts("out1 ---");
        dmatrix_print(&out1);
        puts("out2 ---");
        dmatrix_print(&out2);
        puts("-------");
    }

}
