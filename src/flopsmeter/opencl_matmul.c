/*
**  opencl_matmul.c
**
** implements only the opencl_matmul_t() function.
**
**
** Copyright by Maulis, Adam, 2025 in terms of AGPL v3 or newer
**
*/

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#include "opencl_helper.h"
#include "opencl_matmul.h"

// example for modulo 5: 0->0; 1->5; 4->5; 5->5; 6->10
static inline size_t roundup(size_t what, size_t modulo)
{
    return what + (modulo - what % modulo) % modulo;
}

static void _assert_malloc(void * ptr, const char * file, const int line)
{
    if(NULL == ptr){
        dprintf(2, "Cannot allocate memory at %s:%d\n", file, line);
        exit(2);
    }
}
#define assert_malloc(a) _assert_malloc(a, __FILE__, __LINE__)


static const char ** opencl_readsource(const char * filename)
{
    int infile;
    struct stat statbuff;
    char ** retval;

    infile = open(filename, O_RDONLY);
    if( -1 == infile){
        perror("Cannot open the opencl source file");
        exit(2);
    }
    if( -1 == fstat(infile, &statbuff)){
        perror("Cannot determine the size of the opencl source file");
        exit(2);
    }

    retval = (char **) malloc(sizeof(char*));
    assert_malloc(retval);
    retval[0] = (char*) malloc(sizeof(char) * statbuff.st_size+1);
    assert_malloc(retval[0]);
    if( -1 == read(infile, retval[0], statbuff.st_size)){
        perror("Cannot read the opencl source file");
        exit(2);
    }
    close(infile);
    retval[0][statbuff.st_size] = 0;
    //printf("DEBUG opencl_readsource retval=%p retval[0]=%p\n»%s«\n", retval, retval[0], retval[0]);
    return (const char **) retval;
}


void opencl_initialize_environment(
        cl_device_id device_id,
        cl_command_queue * commandq_out,
        cl_program * program_out
    )
{
    cl_int errval;                     // most cl* functions returns a common status value

    cl_context context;
    cl_command_queue commands;
    cl_program program;

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &errval);
    opencl_assert(errval, "clCreateContext");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, opencl_readsource("opencl_matmul.cl") , NULL, &errval);
    opencl_assert(errval, "clCreateProgramWithSource");

    // Build the program executable
    errval = clBuildProgram(// Builds (compiles and links) a program executable
            program,        // cl_program program,
            1,              // cl_uint num_devices,
            &device_id,     // const cl_device_id* device_list,
#ifdef ROWPADDING
            "-DROWPADDING",
#else
            NULL,           // const char* options, like command line options for gcc, like "-I /tmp -DFOOBAR"
#endif
            NULL,           // pointer of a callback routine for assync build
            NULL);          // void*,  parameter for callback routine
    if (errval != CL_SUCCESS)
    {
        size_t len;
        char * buffer;

        printf("Error in clBuildProgram. Error:%s\n", opencl_GetErrorString(errval));
        printf("Build LOG:\n");
        errval = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        opencl_assert(errval, "clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG-len)");
        buffer = (char  *)malloc(len);
        if( NULL == buffer){
            dprintf(2, "Error allocating memory to print the log of the failed build.\n");
            exit(2);
        }
        errval = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        opencl_assert(errval, "clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG)");
        puts(buffer);
        exit(2);
    }

    // Create a command Q
    commands = clCreateCommandQueue(context, device_id, 0, &errval);
    opencl_assert(errval, "clCreateCommandQueue");

    *commandq_out = commands;
    *program_out = program;

}/* end of opencl_initialize_environment */

#define matrix_t dmatrix
#define PREC_name(a) d##a
#define PREC_t double
#define PREC_kernelname "dmatmul_t"
#include "opencl_matmul.inc"
#undef PREC_name
#undef PREC_t
#undef matrix_t
#undef PREC_kernelname
#define matrix_t fmatrix
#define PREC_name(a) f##a
#define PREC_t float
#define PREC_kernelname "fmatmul_t"
#include "opencl_matmul.inc"
