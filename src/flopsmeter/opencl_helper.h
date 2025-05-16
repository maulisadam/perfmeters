/*
** opencl_helper.h
**
** copyright by Maulis, Adam 2025 by terms of AGPL v3 or newer
*/

#ifndef _OPENCL_HELPER_H
#define _OPENCL_HELPER_H

void _opencl_assert(int returncode, const char * message, const char * file, const int line);
#define opencl_assert(a,b) _opencl_assert(a,b, __FILE__, __LINE__)

char * opencl_GetErrorString(cl_int error);

#endif /* _OPENCL_HELPER_H */
