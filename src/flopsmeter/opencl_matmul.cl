/*
** opencl_matmul.cl
**
** imlementation of the matrix multiplication for OpenCL
** purerly for performance measurement reasons
**
** Copiright by Maulis, Adam, terms of GNU AGPL v3 or newer
**
*/

#define PREC_name(a) d##a
#define PREC_t double
#include "opencl_matmul.cl.inc"
#undef PREC_name
#undef PREC_t
#define PREC_name(a) f##a
#define PREC_t float
#include "opencl_matmul.cl.inc"


/* vim: set filetype=c : */
