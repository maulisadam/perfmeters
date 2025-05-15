/* matmul.h
**
** definitions of some matrix functions
**
** purely for perfomance measuerement purposes
**
**
**  Copyright by Adam Maulis, 2025, In terms of GNU AGPL v3 or newer
**
*/



#ifndef __MATMUL_H
#define __MATMUL_H

/* https://stackoverflow.com/questions/69977196/how-to-correctly-determine-at-compile-time-that-float16-is-supported */
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>
#include <stdint.h>




#ifdef USE_RESTRICT
#   define _RESTRICT restrict
#else
#   define _RESTRICT
#endif

#ifdef USE_ROWALIGN
#  define POINTERTYPE **
#else
#  define POINTERTYPE *
#endif

#ifdef USE_ROWALIGN
#  ifndef ROWALIGN
#  define ROWALIGN 16
#  endif
#else
#  ifdef ROWALIGN
#  undef ROWALIGN
#  endif
#endif

#define MATRIX_MAGIC 4294967087u  /* a nice prime. no other meanings */

typedef struct { __float128 POINTERTYPE _RESTRICT m; uint32_t i, j; uint32_t magic; } qmatrix;
#   define qmatrix_init(a,b,c)    q_matrix_init(a,b,c, __FILE__, __LINE__)
#   define qmatrix_load(a,b)      q_matrix_load(a,b, __FILE__, __LINE__)
#   define qmatrix_free(a)        q_matrix_free(a, __FILE__, __LINE__)
#   define qmatrix_random(a)      q_matrix_random(a, __FILE__, __LINE__)
#   define qmatrix_transpose(a,b) q_matrix_transpose(a,b, __FILE__, __LINE__)
#   define qmatrix_print(a)       q_matrix_print(a, __FILE__, __LINE__)
#   define qmatrix_save(a, b)     q_matrix_save(a,b, __FILE__, __LINE__)
#   define qmatmul_t(a,b,c)       q_matmul_t(a,b,c, __FILE__, __LINE__)
void qmatrix_assert(const qmatrix* _RESTRICT dp, const char * msg, const char * file, const int line);
void q_matrix_init(qmatrix* dp, uint32_t i, uint32_t j, const char* file, const int line);
void q_matrix_load (qmatrix* dp, const char * filename, const char* file, const int line);
void q_matrix_free(qmatrix* dp, const char* file, const int line);
void q_matrix_random(qmatrix* dp,  const char * file, const int line);
void q_matrix_transpose(const qmatrix* in, qmatrix* out,  const char * file, const int line);
void q_matrix_print(const qmatrix* dp, const char * file, const int line);
void q_matrix_save(const qmatrix* dp, const char * savefile, const char * file, const int line);
void q_matmul_t(qmatrix* _RESTRICT in1, qmatrix* _RESTRICT in2, qmatrix* _RESTRICT out, const char * file, const int line);

typedef struct { __float80 POINTERTYPE _RESTRICT m; uint32_t i, j; uint32_t magic; } ldmatrix;
#   define ldmatrix_init(a,b,c)    ld_matrix_init(a,b,c, __FILE__, __LINE__)
#   define ldmatrix_load(a,b)      ld_matrix_load(a,b, __FILE__, __LINE__)
#   define ldmatrix_free(a)        ld_matrix_free(a, __FILE__, __LINE__)
#   define ldmatrix_random(a)      ld_matrix_random(a, __FILE__, __LINE__)
#   define ldmatrix_transpose(a,b) ld_matrix_transpose(a,b, __FILE__, __LINE__)
#   define ldmatrix_print(a)       ld_matrix_print(a, __FILE__, __LINE__)
#   define ldmatrix_save(a, b)     ld_matrix_save(a,b, __FILE__, __LINE__)
#   define ldmatmul_t(a,b,c)       ld_matmul_t(a,b,c, __FILE__, __LINE__)
void ldmatrix_assert(const ldmatrix* _RESTRICT dp, const char * msg, const char * file, const int line);
void ld_matrix_init(ldmatrix* dp, uint32_t i, uint32_t j, const char* file, const int line);
void ld_matrix_load (ldmatrix* dp, const char * filename, const char* file, const int line);
void ld_matrix_free(ldmatrix* dp, const char* file, const int line);
void ld_matrix_random(ldmatrix* dp,  const char * file, const int line);
void ld_matrix_transpose(const ldmatrix* in, ldmatrix* out,  const char * file, const int line);
void ld_matrix_print(const ldmatrix* dp, const char * file, const int line);
void ld_matrix_save(const ldmatrix* dp, const char * savefile, const char * file, const int line);
void ld_matmul_t(ldmatrix* _RESTRICT in1, ldmatrix* _RESTRICT in2, ldmatrix* _RESTRICT out, const char * file, const int line);

typedef struct { double POINTERTYPE _RESTRICT m; uint32_t i, j; uint32_t magic; } dmatrix;
#   define dmatrix_init(a,b,c)    d_matrix_init(a,b,c, __FILE__, __LINE__)
#   define dmatrix_load(a,b)      d_matrix_load(a,b, __FILE__, __LINE__)
#   define dmatrix_free(a)        d_matrix_free(a, __FILE__, __LINE__)
#   define dmatrix_random(a)      d_matrix_random(a, __FILE__, __LINE__)
#   define dmatrix_transpose(a,b) d_matrix_transpose(a,b, __FILE__, __LINE__)
#   define dmatrix_print(a)       d_matrix_print(a, __FILE__, __LINE__)
#   define dmatrix_save(a, b)     d_matrix_save(a,b, __FILE__, __LINE__)
#   define dmatmul_t(a,b,c)       d_matmul_t(a,b,c, __FILE__, __LINE__)
void dmatrix_assert(const dmatrix* _RESTRICT dp, const char * msg, const char * file, const int line);
void d_matrix_init(dmatrix* dp, uint32_t i, uint32_t j, const char* file, const int line);
void d_matrix_load (dmatrix* dp, const char * filename, const char* file, const int line);
void d_matrix_free(dmatrix* dp, const char* file, const int line);
void d_matrix_random(dmatrix* dp,  const char * file, const int line);
void d_matrix_transpose(const dmatrix* in, dmatrix* out,  const char * file, const int line);
void d_matrix_print(const dmatrix* dp, const char * file, const int line);
void d_matrix_save(const dmatrix* dp, const char * savefile, const char * file, const int line);
void d_matmul_t(dmatrix* _RESTRICT in1, dmatrix* _RESTRICT in2, dmatrix* _RESTRICT out, const char * file, const int line);

typedef struct { float POINTERTYPE _RESTRICT m; uint32_t i, j; uint32_t magic; } fmatrix;
#   define fmatrix_init(a,b,c)    f_matrix_init(a,b,c, __FILE__, __LINE__)
#   define fmatrix_load(a,b)      f_matrix_load(a,b, __FILE__, __LINE__)
#   define fmatrix_free(a)        f_matrix_free(a, __FILE__, __LINE__)
#   define fmatrix_random(a)      f_matrix_random(a, __FILE__, __LINE__)
#   define fmatrix_transpose(a,b) f_matrix_transpose(a,b, __FILE__, __LINE__)
#   define fmatrix_print(a)       f_matrix_print(a, __FILE__, __LINE__)
#   define fmatrix_save(a, b)     f_matrix_save(a,b, __FILE__, __LINE__)
#   define fmatmul_t(a,b,c)       f_matmul_t(a,b,c, __FILE__, __LINE__)
void fmatrix_assert(const fmatrix* _RESTRICT dp, const char * msg, const char * file, const int line);
void f_matrix_init(fmatrix* dp, uint32_t i, uint32_t j, const char* file, const int line);
void f_matrix_load (fmatrix* dp, const char * filename, const char* file, const int line);
void f_matrix_free(fmatrix* dp, const char* file, const int line);
void f_matrix_random(fmatrix* dp,  const char * file, const int line);
void f_matrix_transpose(const fmatrix* in, fmatrix* out,  const char * file, const int line);
void f_matrix_print(const fmatrix* dp, const char * file, const int line);
void f_matrix_save(const fmatrix* dp, const char * savefile, const char * file, const int line);
void f_matmul_t(fmatrix* _RESTRICT in1, fmatrix* _RESTRICT in2, fmatrix* _RESTRICT out, const char * file, const int line);

#ifdef FLT16_MIN
typedef struct { __float16 POINTERTYPE _RESTRICT m; uint32_t i, j; uint32_t magic; } hmatrix;
#   define hmatrix_init(a,b,c)    h_matrix_init(a,b,c, __FILE__, __LINE__)
#   define hmatrix_load(a,b)      h_matrix_load(a,b, __FILE__, __LINE__)
#   define hmatrix_free(a)        h_matrix_free(a, __FILE__, __LINE__)
#   define hmatrix_random(a)      h_matrix_random(a, __FILE__, __LINE__)
#   define hmatrix_transpose(a,b) h_matrix_transpose(a,b, __FILE__, __LINE__)
#   define hmatrix_print(a)       h_matrix_print(a, __FILE__, __LINE__)
#   define hmatrix_save(a, b)     h_matrix_save(a,b, __FILE__, __LINE__)
#   define hmatmul_t(a,b,c)       h_matmul_t(a,b,c, __FILE__, __LINE__)
void hmatrix_assert(const hmatrix* _RESTRICT dp, const char * msg, const char * file, const int line);
void h_matrix_init(hmatrix* dp, uint32_t i, uint32_t j, const char* file, const int line);
void h_matrix_load (hmatrix* dp, const char * filename, const char* file, const int line);
void h_matrix_free(hmatrix* dp, const char* file, const int line);
void h_matrix_random(hmatrix* dp,  const char * file, const int line);
void h_matrix_transpose(const hmatrix* in, hmatrix* out,  const char * file, const int line);
void h_matrix_print(const hmatrix* dp, const char * file, const int line);
void h_matrix_save(const hmatrix* dp, const char * savefile, const char * file, const int line);
void h_matmul_t(hmatrix* _RESTRICT in1, hmatrix* _RESTRICT in2, hmatrix* _RESTRICT out, const char * file, const int line);
#endif



#endif /* __MATMUL_H */
