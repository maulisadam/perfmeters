/*
**  opencl_matmul.inc
**
** implements only the opencl_matmul_t() function.
**
**
** Copyright by Maulis, Adam, 2025 in terms of AGPL v3 or newer
**
*/

void PREC_name(_opencl_matmul_t)(
    cl_command_queue commands,  // van benne referencia a context-re és a device_id -re
    cl_program program,         // van benne referencia a context-re és a device_id[] -re
    size_t loc1, size_t loc2,   // work group sizes for debug
    matrix_t* _RESTRICT in1,
    matrix_t* _RESTRICT in2,
    matrix_t* _RESTRICT out,
    const char * file, const int line)
{

    cl_int errval;                     // most cl* functions returns a common status value
    size_t global_sizes[2];            // global work sizes (we have 2 dimensions)
    size_t local_sizes[2];             // local work sizes (work_group_size)
    size_t work_group_size;            // CL_KERNEL_WORK_GROUP_SIZE
    size_t work_group_multiplier_hint; // CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE

    cl_context context;                // get this via clGetCommandQueueInfo(CL_QUEUE_CONTEXT)
    cl_device_id device_id;            // get this via clGetCommandQueueInfo(CL_QUEUE_DEVICE)
    cl_kernel kernel;

    cl_mem in1_mem;                     // device memory used for the in1 matrix
    cl_mem in2_mem;                     // device memory used for the in2 matrix
    cl_mem out_mem;                     // device memory used for the out matrix

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

    // get context and device_id
    errval = clGetCommandQueueInfo(commands, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
    opencl_assert(errval, "clGetCommandQueueInfo(CL_QUEUE_CONTEXT)");
    errval = clGetCommandQueueInfo(commands, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, NULL);
    opencl_assert(errval, "clGetCommandQueueInfo(CL_QUEUE_DEVICE)");

    // get the compute kernel <- a function from a program
    kernel = clCreateKernel(program, PREC_kernelname, &errval);
    opencl_assert(errval, "clCreateKernel");

    // EnqueueNDRangeKernel optimizations
    errval = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
    opencl_assert(errval, "clGetKernelWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE)");
    //printf("   maximum work group size: %lu\n", work_group_size);
    errval = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &work_group_multiplier_hint, NULL);
    opencl_assert(errval, "clGetKernelWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)");
    //printf("   work group multiplier hint: %lu\n", work_group_multiplier_hint);

    // We have 2d data set
    // 1) global_sizes must be greater than I,J
    // 2) global sizes must be multiple of local_sizes
    // 3) local_sizes must be multiple of work_group_multiplier_hint
    //    ? in each dimension? the product would make more sense
    if( 0==loc1 || 0==loc2){ // DEBUG/testing
        local_sizes[0] = 1;  // always work but ineffective
        local_sizes[1] = 1;
        for(size_t tryon_multiplier=1; tryon_multiplier < 9; tryon_multiplier++){
            size_t l1, l2;
            //printf("    tryon_multiplier=%lu\n", tryon_multiplier);

            l1 = (size_t) ceil(sqrt( in2->i * tryon_multiplier * work_group_multiplier_hint/ (double)in2->i ));
            l2 = (tryon_multiplier * work_group_multiplier_hint) /  l1;
            if( l1*l2 <= work_group_size){
                local_sizes[0] = l1;
                local_sizes[1] = l2;
                //printf("    probing local sizes: (%lu, %lu)\n", local_sizes[0], local_sizes[1]);
            }

        }
    } else {
        local_sizes[0] = loc1;
        local_sizes[1] = loc2;
    }

    if( local_sizes[0] * local_sizes[1] > work_group_size){
        //printf("   Given workgroup (%lu, %lu ) is greather than %lu do nothing\n", local_sizes[0], local_sizes[1], work_group_size);
        return ;
    }

    //printf("   Choosed local sizes: (%lu, %lu)\n", local_sizes[0], local_sizes[1]);
    /* per-dimensional roundup*/
    global_sizes[0] = roundup(in1->i, local_sizes[0]); // I
    global_sizes[1] = roundup(in2->i, local_sizes[1]); // J
    //printf("   Choosed global sizes: (%lu, %lu)\n", global_sizes[0], global_sizes[1]);

    // Create buffers in device memory. Also copy in immediatly
#ifdef ROWPADDING
    in1_mem = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(PREC_t) * in1->i * roundup(in1->j, ROWPADDING), in1->m, &errval);
    opencl_assert(errval, "clCreateBuffer(in1)");
    in2_mem = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(PREC_t) * in2->i * roundup(in2->j, ROWPADDING), in2->m, &errval);
    opencl_assert(errval, "clCreateBuffer(in2)");
    out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(PREC_t) * out->i * roundup(out->j, ROWPADDING), NULL, &errval);
    opencl_assert(errval, "clCreateBuffer(out)");
#else
    in1_mem = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(PREC_t) * in1->i * in1->j, in1->m, &errval);
    opencl_assert(errval, "clCreateBuffer(in1)");
    in2_mem = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(PREC_t) * in2->i * in2->j, in2->m, &errval);
    opencl_assert(errval, "clCreateBuffer(in2)");
    out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(PREC_t) * out->i * out->j, NULL, &errval);
    opencl_assert(errval, "clCreateBuffer(out)");
#endif

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
#ifdef ROWPADDING
    {
        uint32_t roundup_j = roundup(in2->i, ROWPADDING);
        uint32_t roundup_y = roundup(in1->j, ROWPADDING);
        errval = clSetKernelArg(kernel, 6, sizeof(uint32_t), &(roundup_j));
        opencl_assert(errval, "clSetKernelArg(6=Jroundup");
        errval = clSetKernelArg(kernel, 7, sizeof(uint32_t), &(roundup_y));
        opencl_assert(errval, "clSetKernelArg(7=Yroundup");
    }
#endif


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

    /* unecessary: see clEnqueueReadBuffer( CL_TRUE)
        // Wait for the command commands to get serviced before reading back results
        errval = clFinish(commands);
        opencl_assert(errval, "clFinish");
    */

    // Read back the results from the device to verify the output
    errval = clEnqueueReadBuffer(
            commands,
            out_mem,
            CL_TRUE,        // blocking read. This calls an implicit clFinish
            0,              // offset
#ifdef ROWPADDING
            sizeof(PREC_t) * out->i * roundup(out->j, ROWPADDING),
#else
            sizeof(PREC_t) * out->i * out->j,
#endif
            out->m,
            0, NULL, NULL); // events parameters: we are not event-based
    opencl_assert(errval, "clEnqueueReadBuffer");

    clReleaseMemObject(in1_mem);
    clReleaseMemObject(in2_mem);
    clReleaseMemObject(out_mem);
    clReleaseKernel(kernel);

} /* end of opencl_matmul_t */

/* vim: set filetype=c : */
