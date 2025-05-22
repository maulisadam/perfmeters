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
    __global PREC_t* in1,  /* matrix, shape=(I,K) */
    __global PREC_t* in2,  /* matrix, shape=(J,K) */
    __global PREC_t* out,  /* matrix, shape=(I,J) */
    const unsigned int I,
    const unsigned int J,
    const unsigned int K
#ifdef ROWPADDING
    ,
    const unsigned int Jroundup,
    const unsigned int Kroundup
#endif
    )
{
    unsigned int kk __attribute__((aligned(8)));
    PREC_t retval __attribute__((aligned(8)));
    unsigned int ii __attribute__((aligned(8))) = get_global_id(0);
    unsigned int jj __attribute__((aligned(8))) = get_global_id(1);
    __global PREC_t * rowpointer1 __attribute__((aligned(8)));
    __global PREC_t * rowpointer2 __attribute__((aligned(8)));

#ifdef ROWPADDING
    if( ii < I && jj < J){
        rowpointer1 = in1 + ii*Kroundup;
        rowpointer2 = in2 + jj*Kroundup;
        retval = dot(vload4(0, rowpointer1), vload4(0, rowpointer2));
        //printf("DEBUG work ii:%u jj:%u initial retval:%f\n", ii,jj, retval);
        for(kk=1; kk < K/4; kk++){
            retval +=  dot(vload4(kk, rowpointer1), vload4(kk, rowpointer2));
            //printf("    DEBUG inside loop kk:%u retval:%f\n", kk, retval);
        }
        //printf("    DEBUG final (%u, %u) retval:%f\n", ii, jj, retval);
        out[ii * Jroundup + jj] = retval;
    }
#else
    if( ii < I && jj < J){
        rowpointer1 = in1 + ii*K;
        rowpointer2 = in2 + jj*K;
        retval = rowpointer1[0] * rowpointer2[0];
        for(kk=1; kk < K; kk++){
            retval += rowpointer1[kk] * rowpointer2[kk];
        }
        out[ii * J + jj] = retval;
    }
#endif
}

/* vim: set filetype=c : */
