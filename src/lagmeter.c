/* lagmeter.c
**
**	Author: Adam Maulis
**	2012.12.17 
**	Copyright: GNU GPL v3 or newer
**
**
**	Description: Ctrl-c -> exit & print the maximum lag.
**
**	Build notes:
**	gcc -O3 -lpthread -o lagmeter lagmeter.c
*/

#define _GNU_SOURCE
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <pthread.h>
#include <errno.h>
#include "meeting.h"

/*
** global variables
*/

static struct timeval maxdelta, mindelta, orig, now, old, delta;
static volatile long long unsigned int cnt;
static int debug, affinity;
static meeting_t presleepmeet, aftersleepmeet;

/*
** some helper functions
*/


void help(void)
{
	puts("Lag measurement.");
	puts("\tUsage: lagmeter [-d] [-a] [-h]");
	puts("\tHit ctrl-c when ready");
	puts("\t\t-d debug messages (-dd for more)");
	puts("\t\t-h this help.");
	puts("\t\t-a cpu affinity: binds each thread to an uniq cpu.");
	
}


void printdata(void)
{
	double avg;

	avg= (double) now.tv_sec + (double) now.tv_usec/1000000.0 -
	     ((double) orig.tv_sec + (double) orig.tv_usec/1000000.0 ) ;
	avg = avg / (double)cnt;

	printf("LAG max: %d.%06d avg: %1.6f min: %d.%06d\n",
			maxdelta.tv_sec, maxdelta.tv_usec, 
			avg,
			mindelta.tv_sec, mindelta.tv_usec);
	fflush(stdout);
}

int ctrlchandler(int sig)
{
	signal(SIGINT, SIG_IGN);
	printdata();
	exit(0);
}

/*
**  cpucnt()  - returns the avaiable cpu count.
*/

int cpucnt()
{
	int ncpus;
	int avaiable_cpu_count;
	size_t setsize;
	cpu_set_t *setp;
	int status;

	ncpus = sysconf(_SC_NPROCESSORS_ONLN);
	if ( ncpus <1 || ncpus > 1000000 ){
		fprintf(stderr,"Err: Invalid number of cpus: %d\n",ncpus);
		fflush(stderr);
		return -1;
	}

	setsize = CPU_ALLOC_SIZE( ncpus);

	setp = CPU_ALLOC(ncpus);
	if( NULL == setp){
		fprintf(stderr,"Err: Cannot allocate memory for cpuset\n");
		fflush(stderr);
		return -1;
	}

	status = sched_getaffinity( 0, setsize, setp);
	if( 0 != status){
		perror("Err: sched_getaffinity said");
		fflush(stderr);
		return -1;
	}

	avaiable_cpu_count = CPU_COUNT_S(setsize, setp);

	if ( 1 < debug ){
		int i;
		printf("Debug: Processors online:          %d\n", ncpus);
		printf("Debug: Bytes required for cpuset:  %d\n", setsize);
		printf("Debug: Processors avaiable:        %d\n", avaiable_cpu_count);
		printf("Debug: cpuset: (hex) ");
		for( i=0; i<setsize; i++){
			printf(" %02X", ((unsigned char *)setp)[i]);
		}
		printf("\n");
		fflush(stdout);
	}

	CPU_FREE(setp);
	return avaiable_cpu_count;
}

/*
** cpuaffinity() - sets the cpu affinity
**	result: cpu number to set to.
*/
int cpuaffinity(int relcpu)
{
	size_t setsize;
	cpu_set_t *setp;
	int status, cpuselect, cpucounting, ncpus;


	ncpus = sysconf(_SC_NPROCESSORS_ONLN);
	if ( ncpus <1 || ncpus > 1000000 ){
		fprintf(stderr,"Err: Invalid number of cpus: %d\n",ncpus);
		exit(1);
	}

	setsize = CPU_ALLOC_SIZE( ncpus);

	setp = CPU_ALLOC(ncpus);
	if( NULL == setp){
		fprintf(stderr,"Err: Cannot allocate memory for cpuset\n");
		exit(1);
	}

	status = sched_getaffinity( 0, setsize, setp);
	if( 0 != status){
		perror("Err: sched_getaffinity said");
		exit(1);
	}

	/*
	** wich one is a relcpu ?
	*/

	cpuselect = 0;
	while( 0 == CPU_ISSET_S(cpuselect, setsize, setp) ) 
		cpuselect++;  /* step to the first avail cpu */

	for( cpucounting=0; cpucounting < relcpu; cpucounting ++){ /* step relcpu times */
		cpuselect++; /* leave the previous avail cpu */
		while( 0 == CPU_ISSET_S(cpuselect, setsize, setp) ) 
			cpuselect++; /* step to the next avail cpu */
	}

	/* 
	** and now, the right cpu number in the variable 'cpuselect' 
	*/

	CPU_ZERO_S(setsize, setp);
	if( 1 < debug){
		printf("Debug: setting cpu affinity to %d%s cpu: %d\n",
			relcpu, 
			(1==relcpu) ? "st" : ((2==relcpu)? "nd" : ((3==relcpu)?"rd" : "th")) , 
			cpuselect );
	}
	CPU_SET_S(cpuselect, setsize, setp);

	status = sched_setaffinity( 0, setsize, setp);
	if( 0 != status ){
		perror("Err: sched_setaffinity" );
		fflush(stderr);
	}
	CPU_FREE(setp);
	return cpuselect;
}

/*
** master() - the main loop: the program logic
*/

int master ( void)
{
	int status, result;
	pid_t pid, tid;
	int threadid;

	threadid=0;  /* this is a master */
	pid = getpid();
	tid = syscall(SYS_gettid);
	if( affinity ){
		result = cpuaffinity(threadid);
		if( debug ){
			printf("Debug: %d:%d master starts on cpu %d\n", pid,tid, result);
			fflush(stdout);
		}
	} else if( debug) {
		printf("Debug: %d:%d master starts\n", pid,tid); 
		fflush(stdout);
	}


	gettimeofday( &orig, NULL);
	old = orig;
	mindelta = old;

	for(cnt=1 ; ; cnt++){

		/* we MUST ensure that every task leave this meeting point 
		*  before we will meet again here. So we need an another meeting point. 
		*/
		status=meeting_wait(&presleepmeet);
		if( 0 != status ){
			fprintf(stderr, "Err: worker(%d:%d) meeting1 failed\n",pid,tid); 
			fflush(stderr);
		}
		status = usleep(1); 
		if( 0 != status ){
			fprintf(stderr, "Err: usleep() failed\n"); fflush(stderr);
		}

		status=meeting_wait(&aftersleepmeet);
		if( 0 != status ){
			fprintf(stderr, "Err: worker(%d:%d) meeting2 failed\n",pid,tid); 
			fflush(stderr);
		}

		gettimeofday( &now, NULL);

		if( now.tv_usec < old.tv_usec){
			delta.tv_usec = 1000000 + now.tv_usec - old.tv_usec;
			delta.tv_sec = now.tv_sec - old.tv_sec -1;
		} else{
			delta.tv_sec = now.tv_sec - old.tv_sec;
			delta.tv_usec = now.tv_usec - old.tv_usec;
		}
		if( ( delta.tv_sec > maxdelta.tv_sec ) || 
		    (( delta.tv_sec == maxdelta.tv_sec ) && 
                    (delta.tv_usec > maxdelta.tv_usec) ) ) {
			maxdelta = delta;
		}

		if( ( delta.tv_sec < mindelta.tv_sec ) || 
		    (( delta.tv_sec == mindelta.tv_sec ) && 
                    (delta.tv_usec < mindelta.tv_usec) ) ) {
			mindelta = delta;
		}
		old = now;

	}
}

/*
**  worker() - subthreads
*/

void * worker( void * param)
{
	int status, result;
	pid_t pid, tid;
	int threadid;
	
	threadid=* ((int*)param);

	pid = getpid();
	tid = syscall(SYS_gettid);

	if( affinity ){
		result = cpuaffinity(threadid);
		if( debug) {
			printf("Debug: %d:%d worker(%d) starts on cpu %d\n", 
			pid,tid, threadid, result);
			fflush(stdout);
		}
	} else if( debug) {
		printf("Debug: %d:%d worker(%d) starts\n", pid,tid, threadid); 
		fflush(stdout);
	}



	for( ; ; ){

		status=meeting_wait(&presleepmeet);
		if( 0 != status ){
			fprintf(stderr, "Err: worker(%d) meeting failed\n",threadid); 
			fflush(stderr);
		}

/*		 status = usleep(1);
		if( 0 != status ){
			fprintf(stderr, "Err: worker(%d) usleep() failed\n",threadid); 
			fflush(stderr);
		}
*/
		status=meeting_wait(&aftersleepmeet);
		if( 0 != status ){
			fprintf(stderr, "Err: worker(%d) meeting2 failed\n",threadid); 
			fflush(stderr);
		}
	}
}

/*
**  main() - the entry point & initialization
*/


int main(int argc, char*argv[])
{
	pthread_t *threads;
	int *params;
	int ncpus, i, status, opt;

	delta.tv_sec=0;
	delta.tv_usec=0;
	maxdelta.tv_sec=0;
	maxdelta.tv_usec=0;
	debug = 0;
	affinity = 0; 


	for( opt = getopt(argc, argv, "dha") ; -1 != opt; opt = getopt(argc, argv, "dha") ){
		switch( opt){
			case 'h':
			case '?':	help();
					return 1;
			case 'd':	debug++;
					break;
			case 'a':	affinity=1;
					break;
		}/* switch opt */
	} /* for opt */


	signal(SIGINT, (sighandler_t) ctrlchandler);

	ncpus = cpucnt();
	if ( ncpus <1 || ncpus > 1000000 ){
		fprintf(stderr,"Err: cannot determine the number of avaiable cpus\n");
		return 1;
	}


	if( debug) {
		printf("Debug: Number of available cpus: %d. Starting %d workers.\n",
		ncpus, ncpus-1); 
		fflush(stdout);
	}

	status = meeting_init( &presleepmeet, ncpus );
	if( 0 != status ) {
		fprintf(stderr,"Err: thread meeting initialization error\n");
		return 1;
	}
	status = meeting_init( &aftersleepmeet, ncpus );
	if( 0 != status ) {
		fprintf(stderr,"Err: thread meeting initialization error\n");
		return 1;
	}

	threads = (pthread_t *) malloc(sizeof(pthread_t) * (ncpus-1));
	if( NULL == threads && 0 > ncpus-1){
		fprintf(stderr,"Err: malloc(%d) failed\n",sizeof(pthread_t)*(ncpus-1));
		return 1;
	}
	params = (int *) malloc(sizeof(int) * (ncpus-1));
	if( NULL == params && 0 > ncpus-1){
		fprintf(stderr,"Err: malloc(%d) failed\n",sizeof(int)*(ncpus-1));
		return 1;
	}


	for( i=0; i < ncpus-1; i++){
		params[i]=i+1;  /* the zero is a master. Workers: 1...ncpus-1 */
		status = pthread_create( threads+i, 
				NULL, 		/* attributes */
				worker, 	/* function pointer */
				params+i);	/* function parameter */
		if( 0 != status ){
			fprintf(stderr,"Err: pthread_create(%d) is return an error:%d (%s)\n", 
				i, status, strerror(status));
			return 1;
		}
	} /* for i */
	
	master();

	return 0;
}
