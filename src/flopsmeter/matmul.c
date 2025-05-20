/* matmul.c
**
** implement some matrix functions in various precisions
**
** purely for perfomance measuerement purposes
**
**
**  Copyright by Adam Maulis, 2025  In terms of GNU AGPL v3 or newer
**
**
*/


#include "matmul.h"

#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pthread.h>
#include <quadmath.h>

// example for modulo 5: 0->0; 1->5; 4->5; 5->5; 6->10
__attribute__ ((const)) static inline size_t roundup(size_t what, size_t modulo) // const means side effect-free
{
    return what + (modulo - what % modulo) % modulo;
}


#define PRECISION 128
#include "matmul.inc"
#undef PRECISION
#define PRECISION 80
#include "matmul.inc"
#undef PRECISION
#define PRECISION 64
#include "matmul.inc"
#undef PRECISION
#define PRECISION 32
#include "matmul.inc"
#undef PRECISION
#ifdef FLT16_MIN
#  define PRECISION 16
#  include "matmul.inc"
#  undef PRECISION
#endif

