/* matmul.inc
**
** implementation of the matrix multiplication 
**
** purely for perfomance measuerement purposes
**
**
**  Copyright by Adam Maulis, 2025, In terms of GNU GPL v3 or newer
**
**
**  this is a C include file (not a header one). Usage:
**  [...]
**  #define PRECISION 64
**  #include "matmul.inc"
**  #define PRECISION 32
**  #include "matmul.inc"
**
*/





#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pthread.h>


#if PRECISION == 80
#   define PREC_t __float80
#   define PREC_name(x) ld##x
#   define SAVE_FMT "%.22Le%s"
#   define matrix_t ldmatrix
#   define ldmatrix_init(a,b,c)    ld_matrix_init(a,b,c, __FILE__, __LINE__)
#   define ldmatrix_load(a,b)      ld_matrix_load(a,b, __FILE__, __LINE__)
#   define ldmatrix_free(a)        ld_matrix_free(a, __FILE__, __LINE__)
#   define ldmatrix_random(a)      ld_matrix_random(a, __FILE__, __LINE__)
#   define ldmatrix_transpose(a,b) ld_matrix_transpose(a,b, __FILE__, __LINE__)
#   define ldmatrix_print(a)       ld_matrix_print(a, __FILE__, __LINE__)
#   define ldmatrix_save(a, b)     ld_matrix_save(a,b, __FILE__, __LINE__)
#   define ldmatmul_t(a,b,c)       ld_matmul_t(a,b,c, __FILE__, __LINE__)
#elif PRECISION == 64
#   define PREC_t double
#   define PREC_name(x) d##x
#   define SAVE_FMT "%.18e%s"
#   define matrix_t dmatrix
#   define dmatrix_init(a,b,c)    d_matrix_init(a,b,c, __FILE__, __LINE__)
#   define dmatrix_load(a,b)      d_matrix_load(a,b, __FILE__, __LINE__)
#   define dmatrix_free(a)        d_matrix_free(a, __FILE__, __LINE__)
#   define dmatrix_random(a)      d_matrix_random(a, __FILE__, __LINE__)
#   define dmatrix_transpose(a,b) d_matrix_transpose(a,b, __FILE__, __LINE__)
#   define dmatrix_print(a)       d_matrix_print(a, __FILE__, __LINE__)
#   define dmatrix_save(a, b)     d_matrix_save(a,b, __FILE__, __LINE__)
#   define dmatmul_t(a,b,c)       d_matmul_t(a,b,c, __FILE__, __LINE__)
#elif PRECISION == 32
#   define PREC_t float
#   define PREC_name(x) f##x
#   define SAVE_FMT "%.10e%s"
#   define matrix_t fmatrix
#   define fmatrix_init(a,b,c)    f_matrix_init(a,b,c, __FILE__, __LINE__)
#   define fmatrix_load(a,b)      f_matrix_load(a,b, __FILE__, __LINE__)
#   define fmatrix_free(a)        f_matrix_free(a, __FILE__, __LINE__)
#   define fmatrix_random(a)      f_matrix_random(a, __FILE__, __LINE__)
#   define fmatrix_transpose(a,b) f_matrix_transpose(a,b, __FILE__, __LINE__)
#   define fmatrix_print(a)       f_matrix_print(a, __FILE__, __LINE__)
#   define fmatrix_save(a, b)     f_matrix_save(a,b, __FILE__, __LINE__)
#   define fmatmul_t(a,b,c)       f_matmul_t(a,b,c, __FILE__, __LINE__)
#elif PRECISION == 16
#   define PREC_t _Float16
#   define PREC_name(x) h##x
#   define SAVE_FMT "%.6e%s"
#   define matrix_t hmatrix
#   define hmatrix_init(a,b,c)    h_matrix_init(a,b,c, __FILE__, __LINE__)
#   define hmatrix_load(a,b)      h_matrix_load(a,b, __FILE__, __LINE__)
#   define hmatrix_free(a)        h_matrix_free(a, __FILE__, __LINE__)
#   define hmatrix_random(a)      h_matrix_random(a, __FILE__, __LINE__)
#   define hmatrix_transpose(a,b) h_matrix_transpose(a,b, __FILE__, __LINE__)
#   define hmatrix_print(a)       h_matrix_print(a, __FILE__, __LINE__)
#   define hmatrix_save(a, b)     h_matrix_save(a,b, __FILE__, __LINE__)
#   define hmatmul_t(a,b,c)       h_matmul_t(a,b,c, __FILE__, __LINE__)
#else
#   error "Only 16, 32, 64 and 80 bit fp precision are supported!"
#endif


#ifdef USE_RESTRICT
#   define _RESTRICT restrict
#else
#   define _RESTRICT
#endif


#define MATRIX_MAGIC 4294967087u  /* a nice prime. no other meanings */


typedef struct {
    PREC_t ** _RESTRICT m;
    uint32_t i, j;
    uint32_t magic;
} PREC_name(matrix);


static inline void PREC_name(matrix_assert) (const matrix_t* _RESTRICT dp, const char * msg, const char * file, const int line)
{
    if( NULL == dp){
        dprintf(2,"Err in %s: not a matrix %s:%d\n", msg, file, line);
        exit(2);
    }
    if( NULL == dp->m ){
        dprintf(2,"Err in %s : null matrix %s:%d\n", msg, file, line);
        exit(2);
    }
    if( NULL == dp->m[0] ){
        dprintf(2,"Err in %s : null matrix row %s:%d\n", msg, file, line);
        exit(2);
    }
    if( MATRIX_MAGIC != dp->magic ){
        dprintf(2,"Err in %s : wrong magic matrix %s:%d\n", msg, file, line);
        exit(2);
    }
    if( 0 == dp->i || 0 == dp->j ){
        dprintf(2,"Err in %s : null size matrix %s:%d\n", msg, file, line);
        exit(2);
    }

}


static inline void PREC_name(matrix_init_assert)(const matrix_t* _RESTRICT dp, const char * msg, const char * file, const int line)
{
    if( NULL == dp){
        dprintf(2,"Err in init: not a matrix %s:%d\n", file, line);
        exit(2);
    }
    if( MATRIX_MAGIC == dp->magic){
        dprintf(2,"Err in init: used matrix, no reinit %s:%d\n", file, line);
        exit(2);
    }
}


#ifndef ROWALIGN
#define ROWALIGN 16
#endif

void PREC_name(_matrix_init)(matrix_t* dp, uint32_t i, uint32_t j, const char* file, const int line)
{
    uint32_t ii;
    uint32_t roundup_j;

    PREC_name(matrix_init_assert)(dp, "init", file, line);

    if( 0==i || 0 == j){
        dprintf(2,"Err in init: invalid matrix_shape:(%u, %u) %s:%d\n", i,j, file, line);
        exit(2);
    }
    dp->m = (PREC_t **) malloc( sizeof(PREC_t *) * i);
    if( NULL == dp->m ){
        dprintf(2,"Err in init: ENOMEM for malloc %s:%d\n", file, line);
        exit(2);
    }

    roundup_j = j;
    if( 0 != (j % ROWALIGN)){
        roundup_j = (j/ROWALIGN +1) * ROWALIGN;
    }

    for(ii=0; ii<i; ii++){
        dp->m[ii] = (PREC_t *) aligned_alloc(ROWALIGN, sizeof(PREC_t) * roundup_j);
        if( NULL == dp->m[ii] ){
            dprintf(2,"Err in init: ENOMEM for row malloc %s:%d\n", file, line);
            exit(2);
        }
    }
    dp->i = i;
    dp->j = j;
    dp->magic = MATRIX_MAGIC;
}


/*
** Load file saved as python's numpy module:
**
** >>> np.savetxt("B.txt", B)
*/
void PREC_name(_matrix_load) (matrix_t* dp, const char * filename, const char* file, const int line)
{

    int fd, retval, height, width, i, j, old_ii, ii;
    char * buff;
    struct stat statbuf;

    //puts("D load beg");
    PREC_name(matrix_init_assert)(dp, "load", file, line);
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
    PREC_name(_matrix_init)(dp, height, width, file, line);
    //puts("D load after init");

    /* convert in-memory text to matrix_t*/
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


void PREC_name(_matrix_free)(matrix_t* dp, const char* file, const int line)
{
    uint32_t ii;

    PREC_name(matrix_assert)(dp, "free", file, line);
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


/* fill a random */
void PREC_name(_matrix_random)(matrix_t* dp,  const char * file, const int line)
{
    uint32_t ii, jj;

    PREC_name(matrix_assert)(dp, "random", file, line);

    for(ii=0; ii < dp->i; ii++){
        for(jj=0; jj < dp->j; ++jj){
            dp->m[ii][jj] = drand48();
            //dp->m[ii][jj] = ii+jj+1;
        }
    }

}


/* transpose */
void PREC_name(_matrix_transpose)(const matrix_t* in, matrix_t* out,  const char * file, const int line)
{
    uint32_t ii, jj;
    
    PREC_name(matrix_assert)(in, "transpose", file, line);
    PREC_name(matrix_assert)(out, "transpose", file, line);
    if( in->i != out->j || in->j != out->i){
        dprintf(2, "Err in transpose: in (%u,%u) and out(%u,%u) matrix_tshape mismatch %s:%d\n", 
            in->i, in->j, out->i, out->j, file, line);
        exit(2);
    }

    for(ii=0; ii < in->i; ++ii){
        for(jj=0; jj < in->j; ++jj){
            out->m[jj][ii] = in->m[ii][jj];
        }
    }
}


void PREC_name(_matrix_print)(const matrix_t* dp, const char * file, const int line)
{
    uint32_t i,j;

    PREC_name(matrix_assert)(dp, "print", file, line);
    for(i=0; i< dp->i; ++i){
        for(j=0; j< dp->j; ++j){
            printf("%.5f" , (double)dp->m[i][j]);
        }
        puts("");
    }
}


/* numpy.savetxt compatible */
void PREC_name(_matrix_save)(const matrix_t* dp, const char * savefile, const char * file, const int line)
{
    uint32_t i,j;
    FILE * sfd;

    PREC_name(matrix_assert)(dp, "save", file, line);

    sfd = fopen(savefile, "w");
    if( NULL == sfd){
        perror("Error in output save");
        exit(1);
    }
    for(i=0; i< dp->i; ++i){
        for(j=0; j< dp->j-1; ++j){
            fprintf(sfd, SAVE_FMT, dp->m[i][j], " ");
        }
        fprintf(sfd, SAVE_FMT, dp->m[i][j], "\n");
    }
    fclose(sfd);
}


/*******************************************************************
    Lightly optimalized matrix multiplication routine. 
    Requires a second matrix as transposed.
    multithreaded
********************************************************************/

/* this seems faster than cblas_ddot() */
static inline PREC_t PREC_name(vector_dot)(const PREC_t * _RESTRICT v1, const PREC_t * _RESTRICT v2, const uint32_t len)
{
    PREC_t register retval;
    uint32_t i;
     
    retval = *v1 * *v2;
    for(i=1; i < len; i++){
        retval += v1[i]*v2[i];
    }
    return retval;
}


struct PREC_name(matmul_t_threadargs){
    matrix_t* _RESTRICT in1;
    matrix_t* _RESTRICT in2;
    matrix_t* _RESTRICT out;
    uint32_t from;
    uint32_t to;
};


static inline void PREC_name(_matmul_t_sub)(
        const matrix_t* _RESTRICT in1, 
        const matrix_t* _RESTRICT in2, 
        matrix_t* _RESTRICT out, 
        const uint32_t from, 
        const uint32_t to)
{
    uint32_t oi, oj;
    PREC_t * _RESTRICT rowpointer1;
    PREC_t * _RESTRICT rowpointer2;

    for(oi=from; oi < to; oi++){
        rowpointer1 = in1->m[oi];
        for(oj=0; oj<out->j; oj++){
            rowpointer2 = in2->m[oj];
            out->m[oi][oj] = PREC_name(vector_dot)(rowpointer1, rowpointer2, in1->j);
        }
    }
}


static void * PREC_name(_matmul_t_thread)(void *arg)
{
    struct PREC_name(matmul_t_threadargs) * myarg = (struct PREC_name(matmul_t_threadargs) *) arg;
    PREC_name(_matmul_t_sub)( myarg->in1, myarg->in2, myarg->out, myarg->from, myarg->to);
    return NULL;
}


void PREC_name(_matmul_t)(matrix_t* _RESTRICT in1, matrix_t* _RESTRICT in2, matrix_t* _RESTRICT out, const char * file, const int line)
{

    PREC_name(matrix_assert)(in1, "dmatmul_t:in1", file, line);
    PREC_name(matrix_assert)(in2, "dmatmul_t:in2", file, line);
    PREC_name(matrix_assert)(out, "dmatmul_t:out", file, line);

    int threadcnt, i;
    pthread_t * threadp;
    struct PREC_name(matmul_t_threadargs) * threadargs;

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
    threadargs = (struct PREC_name(matmul_t_threadargs) *) malloc( threadcnt * sizeof(struct PREC_name(matmul_t_threadargs)));
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
        retval = pthread_create(&(threadp[i]), 0, &PREC_name(_matmul_t_thread), threadargs+i);
        if( retval ) {
            dprintf(2, "Err: matmul_t thread creation. Error code: %d\n", retval);
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

#undef matrix_t
#undef PREC_name
#undef PREC_t
#undef SAVE_FMT
#undef PRINT_FMT
