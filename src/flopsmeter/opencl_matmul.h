/* opencl_matmul.h
**
** definitions of some matrix functions
**
** purely for perfomance measuerement purposes
**
**
**  Copyright by Adam Maulis, 2025, In terms of GNU AGPL v3 or newer
**
*/



#ifndef __OPENCL_MATMUL_H
#define __OPENCL_MATMUL_H

#include "matmul.h"
#include <CL/opencl.h>


#define opencl_dmatmul_t(a,b,c,d)       opencl_d_matmul_t(a,b,c,d, __FILE__, __LINE__)
void opencl_d_matmul_t(cl_device_id device_id, dmatrix* _RESTRICT in1, dmatrix* _RESTRICT in2, dmatrix* _RESTRICT out, const char * file, const int line);


#endif /* __OPENCL_MATMUL_H */
