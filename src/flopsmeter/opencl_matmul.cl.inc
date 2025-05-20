/*
** opencl_matmul.cl.inc
**
** imlementation of the matrix multiplication for OpenCL
** purerly for performance measurement reasons
**
** Copiright by Maulis, Adam, terms of GNU AGPL v3 or newer
**
*/


__kernel void PREC_name(matmul_t)(
    __global PREC_t* in1,  /* matrix, shape=(I,Y) */
    __global PREC_t* in2,  /* matrix, shape=(J,Y) */
    __global PREC_t* out,  /* matrix, shape=(I,J) */
    const unsigned int I,
    const unsigned int J,
    const unsigned int Y
#ifdef ROWPADDING
    ,
    const unsigned int Jroundup,
    const unsigned int Yroundup
#endif
    )
{
    unsigned int yy __attribute__((aligned(8)));
    PREC_t retval __attribute__((aligned(8)));
    unsigned int ii __attribute__((aligned(8))) = get_global_id(0);
    unsigned int jj __attribute__((aligned(8))) = get_global_id(1);
    __global PREC_t * rowpointer1 __attribute__((aligned(8)));
    __global PREC_t * rowpointer2 __attribute__((aligned(8)));

#ifdef ROWPADDING
    if( ii < I && jj < J){
        rowpointer1 = in1 + ii*Yroundup;
        rowpointer2 = in2 + jj*Yroundup;
        retval = dot(vload4(0, rowpointer1), vload4(0, rowpointer2));
        //printf("DEBUG work ii:%u jj:%u initial retval:%f\n", ii,jj, retval);
        for(yy=1; yy < Y/4; yy++){
            retval +=  dot(vload4(yy, rowpointer1), vload4(yy, rowpointer2));
            //printf("    DEBUG inside loop yy:%u retval:%f\n", yy, retval);
        }
        //printf("    DEBUG final (%u, %u) retval:%f\n", ii, jj, retval);
        out[ii * Jroundup + jj] = retval;
    }
#else
    if( ii < I && jj < J){
        rowpointer1 = in1 + ii*Y;
        rowpointer2 = in2 + jj*Y;
        retval = rowpointer1[0] * rowpointer2[0];
        for(yy=1; yy < Y; yy++){
            retval += rowpointer1[yy] * rowpointer2[yy];
        }
        out[ii * J + jj] = retval;
    }
#endif
}

/* vim: set filetype=c : */
