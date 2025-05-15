/*
** opencl_matmul.cl
**
** imlementation of the matrix multiplication for OpenCL
** purerly for performance measurement reasons
**
** Copiright by Maulis, Adam, terms of GNU AGPL v3 or newer
**
*/

__kernel void fmatmul_t(
    __global double* in1,  /* matrix, shape=(I,Y) */ 
    __global double* in2,  /* matrix, shape=(J,Y) */
    __global double* out,  /* matrix, shape=(I,J) */
    const int I,
    const int J, 
    const int Y)
{
    int yy;
    double retval;
    int ii = get_global_id(0);
    int jj = get_global_id(1);
    __global double * rowpointer1 = in1 + ii*Y;
    __global double * rowpointer2 = in2 + jj*Y;

    if( ii < I && jj < J){
        retval = rowpointer1[0] * rowpointer2[0];
        for(yy=1; yy < Y; yy++){
            retval += rowpointer1[yy] * rowpointer2[yy];
        }
        out[ii * J + jj] = retval;
    }
}

/* vim: set filetype=c : */
