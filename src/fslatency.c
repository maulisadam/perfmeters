/*
** fslatency.c
**
** measure filesystem (disk) write latency for a long period
** tested at Ubuntu 24.04 LTS
**
** gcc -Wall -o fslatency fslatency.c
**
** Copyright by Adam Maulis maulis@ludens.elte.hu 2024

 This program is free software: you can redistribute it and/or modify it under 
 the terms of the GNU Affero General Public License as published by the 
 Free Software Foundation, either version 3 of the License, or 
 (at your option) any later version.

 This program is distributed in the hope that it will be useful, 
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 See the GNU Affero General Public License for more details.

*/

# define _GNU_SOURCE 1

#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>


const struct timespec TENTHSECOND = { 0, 100000000 }; /* sec, nanosec */


/* 
** time differencial utility function: returns int (nanoseconds)
*/
long int diff_timespec(const struct timespec *endtime, const struct timespec *begtime) {
  return (endtime->tv_sec - endtime->tv_sec) * 1000000000L + 
            (endtime->tv_nsec - begtime->tv_nsec);
}

int main()
{
    int fd;
    int retval;
    struct timespec tprecision;
    struct timespec begtime;
    struct timespec endtime;
    long int difftime=0;

    size_t bufflen = 300;
    char buff[bufflen];

    fd = open("./fslatencytestfile.txt", O_WRONLY | O_CREAT | O_EXCL | O_SYNC | O_DSYNC | O_NOATIME );
    if( fd < 0 ){
        perror("cannot create for write ./fslatencytestfile.txt");
        return 1;
    }

    clock_getres(CLOCK_REALTIME, &tprecision);
    printf("Time measuring precision: %ld nanoseconds\n", tprecision.tv_nsec);



    clock_gettime(CLOCK_REALTIME, &begtime);
    retval = write(fd, "wallclock_time_s  prevlatencyns\n", 32 );
    if( retval < 0){
        perror("cannot write header");
        return 1;
    }
    clock_gettime(CLOCK_REALTIME, &endtime);
    difftime = diff_timespec(&endtime, &begtime);

    printf("infinite measuring loop starts. Press ctrl-c when bored\n");
    while(1){
        
        clock_gettime(CLOCK_REALTIME, &begtime);
        //printf("DEBUG new sleep at %ld.%09ld\n", begtime.tv_sec, begtime.tv_nsec);

        /* special 32 byte string (with \n but without treminating zero */
        snprintf(buff, bufflen, "%9ld.%08ld %011ld\n", begtime.tv_sec,begtime.tv_nsec/10, difftime);
        //printf("DEBUG l=%lu \"%s\"\n", strlen(buff), buff);
 
        retval = write(fd, buff, 32);
        if( retval < 0){
            perror("cannot write");
            return 2;
        }
        clock_gettime(CLOCK_REALTIME, &endtime);
        difftime = diff_timespec(&endtime, &begtime);
        retval = nanosleep(&TENTHSECOND, NULL);
        if( retval < 0){
            perror("cannot sleep");
            return 1;
        }
    }

    close(fd);
    return 0;
}
