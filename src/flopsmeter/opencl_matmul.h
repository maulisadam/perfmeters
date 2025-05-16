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

void opencl_initialize_environment(cl_device_id device_id, cl_command_queue * commandq_out, cl_program * program_out);

#define opencl_dmatmul_t(a,b,c,d,e,f,g)       d_opencl_matmul_t(a,b,c,d,e,f,g, __FILE__, __LINE__)
void d_opencl_matmul_t(cl_command_queue commands, cl_program program,
    size_t loc1, size_t loc2,
    dmatrix* _RESTRICT in1, dmatrix* _RESTRICT in2, dmatrix* _RESTRICT out,
    const char * file, const int line);

#define opencl_fmatmul_t(a,b,c,d,e,f,g)       f_opencl_matmul_t(a,b,c,d,e,f,g, __FILE__, __LINE__)
void f_opencl_matmul_t(cl_command_queue commands, cl_program program,
    size_t loc1, size_t loc2,
    fmatrix* _RESTRICT in1, fmatrix* _RESTRICT in2, fmatrix* _RESTRICT out,
    const char * file, const int line);

#endif /* __OPENCL_MATMUL_H */
