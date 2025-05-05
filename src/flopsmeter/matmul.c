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
*/


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>


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
    double * _RESTRICT m;
    uint32_t i, j;
    uint32_t magic;
} dmatrix;

static inline void dmatrix_assert(dmatrix * _RESTRICT dp, const char * msg, const char * file, const int line)
{
    if( NULL == dp){
        dprintf(2,"Err in %s: not a dmatrix %s:%d\n", msg, file, line);
        exit(2);
    }
    if( NULL == dp->m ){
        dprintf(2,"Err in %s : null dmatrix %s:%d\n", msg, file, line);
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

static inline void dmatrix_init_assert(dmatrix * _RESTRICT dp, const char * msg, const char * file, const int line)
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


void _dmatrix_init(dmatrix * dp, uint32_t i, uint32_t j, const char* file, const int line)
{
    dmatrix_init_assert(dp, "init", file, line);

    if( 0==i || 0 == j){
        dprintf(2,"Err in init: invalid dmatrix size(%u, %u) %s:%d\n", i,j, file, line);
        exit(2);
    }
    dp->m = (double *) malloc( sizeof(double) * i * j);
    if( NULL == dp->m ){
        dprintf(2,"Err in init: ENOMEM for malloc %s:%d\n", file, line);
        exit(2);
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

    dmatrix_init_assert(dp, "load", file, line);
    fd = open(filename, O_RDONLY);
    if( fd<0 ){
        perror("Error while opening file:");
        exit(1);
    }
    retval = fstat(fd, &statbuf);
    if( retval <0 ){
        perror("Error while inquire file size:");
        exit(1);
    }
    buff = (char*) malloc(statbuf.st_size);
    if(NULL == buff){
        dprintf(2, "Error in load: no enought memory. %s:%d\n", file, line);
        exit(2);
    }
    retval = read(fd, buff, statbuf.st_size);
    if( retval <0 ){
        perror("Error while file read:");
        exit(1);
    }
    close(fd);
    
    /* determine height(i) and width(j) */
    height = 0;
    for(i=0; i<statbuf.st_size; ++i){
        if('\n' == buff[i]){
            height++;
        }
    }
    width = 1;
    for(ii=0; i<statbuf.st_size; ++ii){
        if('\n' == buff[ii]){
            break;
        }
        if(' ' == buff[ii]){
            width++;
        }
    }
    _dmatrix_init(dp, height, width, file, line);

    /* convert in-memory text to dmatrix */
    old_ii = 0;
    i = j = 0;
    for(ii=0; ii<statbuf.st_size; ++ii){
        if((buff[ii] != '\n') & (buff[ii] != ' ')){
            continue;
        }
        dp->m[i*dp->j + j] = atof(buff+old_ii);
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
#define dmatrix_load(a,b) _dmatrix_load(a,b __FILE__, __LINE__)


void _dmatrix_free(dmatrix * dp, const char* file, const int line)
{
    dmatrix_assert(dp, "free", file, line);
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
    uint32_t ii;

    dmatrix_assert(dp, "random", file, line);

    for(ii=0; ii < dp->i*dp->j; ++ii){
        //dp->m[ii] = drand48();
        dp->m[ii] = ii+1;
    }

}
#define dmatrix_random(a)  _dmatrix_random(a, __FILE__, __LINE__)


/* inline transpose */
void _dmatrix_transpose(dmatrix * dp, const char * file, const int line)
{
    uint32_t oi, oj, tmpi;
    double * tmp;
    

    dmatrix_assert(dp, "transpose", file, line);
    tmp = (double *) malloc( sizeof(double) * dp->i * dp->j);

    for(oi=0; oi < dp->i; ++oi){
        for(oj=0; oj < dp->j; ++oj){
            tmp[oj * dp->i + oi ] = dp->m[oi*dp->j +  oj];
        }
    }
    free(dp->m);
    dp->m = tmp;
    tmpi = dp->i;
    dp->i = dp->j;
    dp->j = tmpi;
}
#define dmatrix_transpose(a)  _dmatrix_transpose(a, __FILE__, __LINE__)


void _dmatrix_print(dmatrix * dp, const char * file, const int line)
{
    uint32_t i,j;

    dmatrix_assert(dp, "print", file, line);
    for(i=0; i< dp->i; ++i){
        for(j=0; j< dp->j; ++j){
            printf("%.5f ", dp->m[ i*dp->j + j ]);
        }
        puts("");
    }
}
#define dmatrix_print(a) _dmatrix_print(a, __FILE__, __LINE__)

/*******************************************************************/


void _dmatmul1(dmatrix * _RESTRICT in1, dmatrix * _RESTRICT in2, dmatrix * _RESTRICT out, const char * file, const int line)
{
    uint32_t opos, oi, oj, ii;

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
            opos = oi*out->j + oj;
            out->m[opos]  = 0.0;
            for(ii=0; ii<in2->i; ++ii){
                 out->m[opos] += in1->m[ oi * (in1->j) + ii ] * in2->m[ ii *in2->j +oj];
            }
        }
    }
}
#define dmatmul1(a,b,c) _dmatmul1(a,b,c, __FILE__, __LINE__)


/*optimalized one */
void _dmatmul2(dmatrix * _RESTRICT in1, dmatrix * _RESTRICT in2, dmatrix * _RESTRICT out, const char * file, const int line)
{
    uint32_t opos, oi, oj, jj;
    double * _RESTRICT rowpointer1;
    double * _RESTRICT rowpointer2;
    register double target;

    dmatrix_assert(in1, "dmatmul:in1", file, line);
    dmatrix_assert(in2, "dmatmul:in2", file, line);
    dmatrix_assert(out, "dmatmul:out", file, line);

    dmatrix_transpose(in2); /* inline! */

    if(in1->i != out->i){
        dprintf(2,"Err: dmatmul: in1 versus out shape mismatch\n");
        exit(2);
    }
    if(in2->i != out->j){
        dprintf(2,"Err: dmatmul: in2 versus out shape mismatch\n");
        exit(2);
    }
    if(in1->j != in2->j){
        dprintf(2,"Err: dmatmul: in1 versus in2 shape mismatch\n");
        exit(2);
    }
    
    for(oi=0; oi < out->i; ++oi){
        rowpointer1 = in1->m + ( oi * (in1->j));
        for(oj=0; oj < out->j; ++oj){
            rowpointer2 = in2->m + ( oj * (in2->j));
            target = 0;
            jj =  in2->j;
            while(jj--){
                 target += rowpointer1[jj] * rowpointer2[jj];
            }
            out->m[ oi*out->j + oj] = target;
        }
    }
}
#define dmatmul2(a,b,c) _dmatmul2(a,b,c, __FILE__, __LINE__)


int main(int argc, const char * argv[])
{
    uint32_t i1, j1, i2, j2;
    struct timespec t1, t2, t3, endtime;
    long int difftime;

    dmatrix in1, in2, out1, out2;

    if( argc != 5){
        dprintf(2,"Usage: ./matmul i1 j1 i2 j2\n");
        dprintf(2,"  i1 j1: shape of first matrix\n");
        dprintf(2,"  i2 j2: shape of second matrix\n");
        return 2;
    }
    clock_gettime(CLOCK_REALTIME, &t1);    
    dmatrix_init(&in1, atoi(argv[1]), atoi(argv[2]));
    dmatrix_init(&in2, atoi(argv[3]), atoi(argv[4]));
    dmatrix_init(&out1, atoi(argv[1]), atoi(argv[4]));
    dmatrix_init(&out2, atoi(argv[1]), atoi(argv[4]));
    dmatrix_random(&in1);
    dmatrix_random(&in2);
    clock_gettime(CLOCK_REALTIME, &t2);

    dmatmul1(&in1, &in2, &out1);
    clock_gettime(CLOCK_REALTIME, &t3);

    dmatmul2(&in1, &in2, &out2);
    clock_gettime(CLOCK_REALTIME, &endtime);

    difftime = diff_timespec(&t2, &t1);
    printf("Initialization time: %f\n", difftime/1000000000.0);
    difftime = diff_timespec(&t3, &t2);
    printf("Matmul (1) time: %f\n", difftime/1000000000.0);
    difftime = diff_timespec(&endtime, &t3);
    printf("Matmul (2) time: %f\n", difftime/1000000000.0);

    if(0){
        puts("in1 -----");
        dmatrix_print(&in1);
        puts("in2 -----");
        dmatrix_print(&in2);
        puts("out1 ---");
        dmatrix_print(&out1);
        puts("out2 ---");
        dmatrix_print(&out2);
        puts("-------");
    }

}
