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

#define matrix_t dmatrix
#define PREC_name(a) d##a

void opencl_d_matmul_t(cl_device_id device_id,
    matrix_t* _RESTRICT in1,
    matrix_t* _RESTRICT in2,
    matrix_t* _RESTRICT out,
    const char * file, const int line)
{

    PREC_name(matrix_assert)(in1, "opencl_matmul_t:in1", file, line);
    PREC_name(matrix_assert)(in2, "opencl_matmul_t:in2", file, line);
    PREC_name(matrix_assert)(out, "opencl_matmul_t:out", file, line);

    if(in1->i != out->i){
        dprintf(2,"Err: opencl_matmul_t: in1 versus out shape mismatch %s:%i\n", file, line);
        exit(2);
    }
    if(in2->i != out->j){
        dprintf(2,"Err: opencl_matmul_t: in2 versus out shape mismatch %s:%i\n", file, line);
        exit(2);
    }
    if(in1->j != in2->j){
        dprintf(2,"Err: opencl_matmul_t: in1 versus in2 shape mismatch %s:%i\n", file, line);
        exit(2);
    }



    cl_int errval;                     // most cl* functions returns a common status value
    size_t global_sizes[2];            // global work sizes (we have 2 dimensions)
    size_t local_sizes[2];             // local work sizes (work_group_size)
    size_t work_group_size;            // CL_KERNEL_WORK_GROUP_SIZE
    size_t work_group_multiplier_hint; // CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE

    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;

    cl_mem in1_mem;                     // device memory used for the in1 matrix
    cl_mem in2_mem;                     // device memory used for the in2 matrix
    cl_mem out_mem;                     // device memory used for the out matrix

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &errval);
    opencl_assert(errval, "clCreateContext");

    // Create a command Q
    commands = clCreateCommandQueue(context, device_id, 0, &errval);
    opencl_assert(errval, "clCreateCommandQueue");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, opencl_readsource("opencl_matmul.cl") , NULL, &errval);
    opencl_assert(errval, "clCreateProgramWithSource");

    // Build the program executable
    errval = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errval != CL_SUCCESS)
    {
        size_t len;
        char buffer[5099]; // meaningless but unique prime as usual

        printf("Error in clBuildProgram. Build LOG:\n");
        errval = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        opencl_assert(errval, "clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG)");
        puts(buffer);
        exit(2);
    }

    // Create a compute kernel <- a function from a program
    kernel = clCreateKernel(program, "fmatmul_t", &errval);
    opencl_assert(errval, "clCreateKernel");

    // Create buffers in device memory. Also copy in immediatly
    in1_mem = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(double) * in1->i * in1->j, in1->m, &errval);
    opencl_assert(errval, "clCreateBuffer(in1)");
    in2_mem = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(double) * in2->i * in2->j, in2->m, &errval);
    opencl_assert(errval, "clCreateBuffer(in2)");
    out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * out->i * out->j, NULL, &errval);
    opencl_assert(errval, "clCreateBuffer(out)");

    // set kernel args
    errval = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in1_mem);
    opencl_assert(errval, "clSetKernelArg(0=in1_mem)");
    errval = clSetKernelArg(kernel, 1, sizeof(cl_mem), &in2_mem);
    opencl_assert(errval, "clSetKernelArg(1=in2_mem)");
    errval = clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem);
    opencl_assert(errval, "clSetKernelArg(2=out_mem)");
    errval = clSetKernelArg(kernel, 3, sizeof(uint32_t), &(in1->i));
    opencl_assert(errval, "clSetKernelArg(3=I)");
    errval = clSetKernelArg(kernel, 4, sizeof(uint32_t), &(in2->i));
    opencl_assert(errval, "clSetKernelArg(4=J)");
    errval = clSetKernelArg(kernel, 5, sizeof(uint32_t), &(in1->j));
    opencl_assert(errval, "clSetKernelArg(5=Y)");

    // EnqueueNDRangeKernel optimizations
    errval = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
    opencl_assert(errval, "clGetKernelWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE)");
    printf("   maximum work group size: %lu\n", work_group_size);
    errval = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &work_group_multiplier_hint, NULL);
    opencl_assert(errval, "clGetKernelWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)");
    printf("   work group multiplier hint: %lu\n", work_group_multiplier_hint);

    // We have 2d data set
    // 1) global_sizes must be greater than I,J
    // 2) global sizes must be multiple of local_sizes
    // 3) local_sizes must be multiple of work_group_multiplier_hint
    //    ? in each dimension? the product would make more sense
    local_sizes[0] = 1;  // always work but ineffective
    local_sizes[1] = 1;
    for(size_t tryon_multiplier=1; tryon_multiplier<8; tryon_multiplier++){
        size_t l1, l2;
        printf("    tryon_multiplier=%lu\n", tryon_multiplier);

        l1 = (size_t) ceil(sqrt( in2->i * tryon_multiplier * work_group_multiplier_hint/ (double)in2->i ));
        l2 = (tryon_multiplier * work_group_multiplier_hint) /  l1;
        if( l1*l2 <= work_group_size){
            local_sizes[0] = l1;
            local_sizes[1] = l2;

            printf("   Choosed local sizes: (%lu, %lu)\n", local_sizes[0], local_sizes[1]);
        }

    }
    global_sizes[0] = roundup(in1->i, local_sizes[0]); // I
    global_sizes[1] = roundup(in2->i, local_sizes[1]); // J
    printf("   Choosed global sizes: (%lu, %lu)\n", global_sizes[0], global_sizes[1]);

    errval = clEnqueueNDRangeKernel(
        commands, //cl_command_queue command_queue,
        kernel,   //cl_kernel kernel,
        2,        //cl_uint work_dim,
        NULL,     //const size_t* global_work_offset,
        global_sizes,  //const size_t* global_work_size,
        local_sizes,   //const size_t* local_work_size,
        0,        //cl_uint num_events_in_wait_list,
        NULL,     //const cl_event* event_wait_list,
        NULL);    //cl_event* event

    opencl_assert(errval, "clEnqueueNDRangeKernel");

    // Wait for the command commands to get serviced before reading back results
    errval = clFinish(commands);
    opencl_assert(errval, "clFinish");

    // Read back the results from the device to verify the output
    errval = clEnqueueReadBuffer(commands, out_mem, CL_TRUE, 0, sizeof(double) * out->i * out->j, out->m, 0, NULL, NULL );
    opencl_assert(errval, "clEnqueueReadBuffer");


    clReleaseMemObject(in1_mem);
    clReleaseMemObject(in2_mem);
    clReleaseMemObject(out_mem);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

} /* end of opencl_matmul_t */

