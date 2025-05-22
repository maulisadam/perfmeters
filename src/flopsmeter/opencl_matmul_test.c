/*
**  opencl_matmul_test.c
**
** Copyright by Maulis, Adam, 2025 in terms of AGPL v3 or newer
**
*/

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#include "opencl_helper.h"
#include "opencl_matmul.h"

/*
** time differencial utility function: returns int (nanoseconds)
*/
static inline long int diff_timespec(const struct timespec *endtime, const struct timespec *begtime) {
  return (endtime->tv_sec - begtime->tv_sec) * 1000000000L +
            (endtime->tv_nsec - begtime->tv_nsec);
}


/*
** in1,in2_t,out is either (fmatrix*) or (dmatrix*)
**  (fmatrix*) when precision==32
*/
static void opencl_calling_part(int precision, void* in1, void*in2_t, void*out)
{
    struct timespec begintime, inittime, endtime;
    double flop;
    /* calculate flops*/
    flop = (double)((fmatrix*)in1)->j * (double)((fmatrix*)in1)->i * (double)((fmatrix*)in2_t)->i;        /* multiplications */
    flop += ((double)((fmatrix*)in1)->j -1) *  (double)((fmatrix*)in1)->i * (double)((fmatrix*)in2_t)->i; /* additions */

    // Opencl calling part

    cl_int err;                            // error code returned from api calls

    cl_platform_id platform_id_list[30];   // compute platform IDs (external loop)
    cl_uint platform_id_list_len;
    char platform_name[300];
    size_t platform_name_len;

    cl_device_id device_id_list[30];       // compute device IDs (internal loop)
    cl_uint device_id_list_len;
    char device_name[300];
    size_t device_name_len;


    // get all platforms and devices
    err = clGetPlatformIDs(30, platform_id_list, &platform_id_list_len);
    opencl_assert(err, "clGetPlatformIDs");
    printf("#This system has %u OpenCL platforms\n", platform_id_list_len);
    for(cl_uint platformindex=0; platformindex < platform_id_list_len; platformindex++){
        err = clGetPlatformInfo(platform_id_list[platformindex], CL_PLATFORM_NAME, 300, platform_name, &platform_name_len);
        opencl_assert(err, "clGetPlatformInfo)CL_PLATFORM_NAME)");
        printf("#    Platform index:%u, id=%p name=%s\n", platformindex, platform_id_list[platformindex], platform_name);

        err = clGetDeviceIDs(
            platform_id_list[platformindex], // cl_platform_id platform, platform refers to the platform ID returned by clGetPlatformIDs
            CL_DEVICE_TYPE_ALL, // cl_device_type device_type, device_type can be used to query specific OpenCL devices or all
                // CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR  = CL_DEVICE_TYPE_ALL
            30, // cl_uint num_entries,
            device_id_list,  // cl_device_id* devices, returns a list of OpenCL devices found
            &device_id_list_len // cl_uint* num_devices
            );
        opencl_assert(err, "clGetDeviceIDs");
        printf("#    This platform has %u OpenCL devices\n", device_id_list_len);
        for(cl_uint deviceindex=0; deviceindex<device_id_list_len;  deviceindex++){
            err = clGetDeviceInfo(device_id_list[deviceindex], CL_DEVICE_NAME, 300, device_name, &device_name_len);
            opencl_assert(err, "clGetDeviceInfo(CL_DEVICE_NAME)");
            printf("#        Device index:%u, id=%p name=%s\n", deviceindex, device_id_list[deviceindex], device_name);
            {
                cl_command_queue commands;
                cl_program program;

                clock_gettime(CLOCK_REALTIME, &begintime);
                opencl_initialize_environment(device_id_list[deviceindex], &commands, &program);
                clock_gettime(CLOCK_REALTIME, &inittime);
                printf("# Precision: fp%d, time_init:%f, total_flop:%f\n",
                    precision,
                    diff_timespec(&inittime, &begintime)/1000000000.0,
                    flop);
                printf("localsize0 localsize1 time_matmul Gflops\n");
                for(size_t loc1=1; loc1<65; loc1++){
                    for(size_t loc2=1; loc2<65; loc2++){
                        clock_gettime(CLOCK_REALTIME, &inittime);
                        if( 32 == precision){
                            opencl_fmatmul_t(commands, program, loc1, loc2, (fmatrix *)in1, (fmatrix *)in2_t, (fmatrix *)out);
                        }else{
                            opencl_dmatmul_t(commands, program, loc1, loc2, (dmatrix *)in1, (dmatrix *)in2_t, (dmatrix *)out);
                        }
                        clock_gettime(CLOCK_REALTIME, &endtime);
                        if( diff_timespec(&endtime, &inittime) > 100000){ // 0.1 msec
                            printf("%lu %lu %f %f\n",
                                loc1, loc2,
                                diff_timespec(&endtime, &inittime)/1000000000.0,
                                flop/diff_timespec(&endtime, &inittime));
                        }
                    }
                }/* end for loc1 */
            }
        }/* end for deviceindex */

    } /* end for platformindex */
}

int main(int argc, const char * argv[])
{

    if( argc != 5){
        dprintf(2,"Usage: ./opencl_matmul_test PREC A B C\n");
        dprintf(2,"\tDo the matrix multiplication where C = A x B");
        dprintf(2,"\tPREC is one of the following: fp16, fp32, fp64, fp80, fp128\n");
        dprintf(2,"\tA and B is existing files on format of numpy.savetxt()\n");
        dprintf(2,"\tC will be same format.\n");

        return 2;
    }

    if( 0 == strcmp(argv[1], "fp32")){
        fmatrix in1, in2, in2_t, out;

        fmatrix_load(&in1, argv[2]);
        fmatrix_load(&in2, argv[3]);
        fmatrix_init(&in2_t, in2.j, in2.i);
        fmatrix_transpose(&in2, &in2_t);

        fmatrix_free(&in2);
        fmatrix_init(&out, in1.i, in2_t.i);

        opencl_calling_part(32, &in1, &in2_t, &out);

        fmatrix_save(&out, argv[4]);
        fmatrix_free(&in1);
        fmatrix_free(&in2_t);
        fmatrix_free(&out);
    } else if( 0 == strcmp(argv[1], "fp64")){
        dmatrix in1, in2, in2_t, out;

        dmatrix_load(&in1, argv[2]);
        dmatrix_load(&in2, argv[3]);
        dmatrix_init(&in2_t, in2.j, in2.i);
        dmatrix_transpose(&in2, &in2_t);

        dmatrix_free(&in2);
        dmatrix_init(&out, in1.i, in2_t.i);

        opencl_calling_part(64, &in1, &in2_t, &out);

        dmatrix_save(&out, argv[4]);
        dmatrix_free(&in1);
        dmatrix_free(&in2_t);
        dmatrix_free(&out);
    }else {
        dprintf(2,"Err: unknown PREC. Must be one of the following: fp32 fp64\n");
        exit(2);
    }



    return 0;
}

