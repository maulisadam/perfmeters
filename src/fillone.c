/*
 * Author: Maulis Adam
 * Description:
 *    file/device filler (performance tool)
 *
 * initial release: 12-jan-2005
 * history:	3-jan-2007, Maulis, blocksize parameter, randomizing
 * 		4-jan-2007, Maulis, output only blocksize - byte/sec
 * 		15-jan-2007, Maulis, tests: {seq,rnd}x{write,read}
 *		21-may-2007, Maulis, O_DIRECT support
 *		18-jul-2007, Maulis, thread support on random tests
 *		11-aug-2007, Maulis, bugfix about buffered seq wr
 *		15-aug-2007, Maulis, different filesize vs datasize
 *		21-aug-2007, Maulis, signed-unsigned bugfixes
 *		 7-sep-2007, Maulis, thread support for seq tests, rewrite rnd tests 
 *		24-sep-2007, Maulis, more confortable DEBUG
 *		17-oct-2016, Maulis, non-compressable (/dev/urandom) 64k fill 
 *		20-nov-2018, Maulis, parameter selectable compressable/noncompressable pattern
 *
 * to build:
 * 
 * apt-get install libaio-dev
 * cc -o fillone -Wall fillone.c -laio

 Copyright by Adam Maulis 2024

 This program is free software: you can redistribute it and/or modify it under 
 the terms of the GNU Affero General Public License as published by the 
 Free Software Foundation, either version 3 of the License, or 
 (at your option) any later version.

 This program is distributed in the hope that it will be useful, 
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 See the GNU Affero General Public License for more details.

 */

#if !defined(__x86_64__)
#error "64 bit architecture only *"
#endif
#define VERS "0.11"  /* update please! */
#define _GNU_SOURCE  /* for O_DIRECT constanst */
#define _LARGEFILE64_SOURCE
#include <inttypes.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/fs.h> /* block device get size ioctl */
#include <libaio.h> /* io_submit/io_setup/io_destroy/io_getevents */

#define MAGIC 0xDEADBEEF
#ifndef TRUE
#define TRUE (0==0)
#endif
#ifndef FALSE
#define FALSE (0!=0)
#endif
#define roundup4k(x) ((((x-1LL)/4096LL)+1LL)*4096LL)

#define RANDOMPOOL "/dev/urandom"

static struct OPT {
	int debug; /* for DEBUG envvar */
	int rawmode;
	int uncompressable;
	int rndfh; /* random pool file handle */
    char * randompool;
	char * fname;
	char * * buff; /* each thread has own buff */
	long long threadcnt;
	unsigned long long mbl;
	unsigned long long datasize;
	unsigned long long filesize;
	unsigned long long totio;
}opt;



long long atoint64(const char *ca)
{
      long long ig=0;
      long long sign=1;
      /* test for prefixing white space */
      while (*ca == ' ' || *ca == '\t' ) 
          ca++;
      /* Check sign entered or no */
      if ( *ca == '-' )       
          sign = -1;
      /* convert string to int */
      while (*ca != '\0')
          if (*ca >= '0' && *ca <= '9')
              ig = ig * 10LL + *ca++ - '0';
          else
              ca++;
      return (ig*sign);
}

unsigned long long getfilesize()
{
	struct stat statit;
	int fd;
	int ret;
	unsigned long int filesize32;
	unsigned long long filesize;
	
	fd=open(opt.fname,O_RDONLY|O_LARGEFILE);
	if( -1 == fd ){
		fprintf(stderr,"Cannot open %s", opt.fname);
		perror(" ");
		exit(1);
	}
	ret = fstat(fd, &statit);
	if( -1 == ret ){
		fprintf(stderr,"Cannot get size of %s", opt.fname);
		perror(" ");
		close(fd);
		exit(1);
	}
	if(S_ISREG(statit.st_mode )){
		filesize = statit.st_size; 
		if(opt.debug){printf(" regural file size: %lld \n", filesize);fflush(stdout);}
	}else if( S_ISBLK(statit.st_mode) ) {

        	ret = ioctl(fd, BLKGETSIZE, &filesize32);
        	if (ret){
			fprintf(stderr,"Cannot get size of %s", opt.fname);
			perror(" ");
                	exit(1);
		}
        	ret = ioctl(fd, BLKGETSIZE64, &filesize);
        	if (ret  || filesize == 0 )
                	filesize = filesize32;

		if(opt.debug){printf("block device size: %lld \n", filesize);fflush(stdout);}
	} else {
		fprintf(stderr,"Unsupported file (not regular nor block device)\n");
		exit(1);
	}
	close(fd);
	return filesize;
}

/*
 * error handling functions
 */
void errh_malloc(void * p, const char * str)
{
	if( NULL == p){
		fprintf(stderr,"Error: insufficient memory to allocate %s\n", str);
		exit(1);
	}
}

void errh_iogeneric(const char * wh, long result)
{
	if( 0 == result ) return;
	switch( result ){
		case -EINVAL: 
			fprintf(stderr, "Error: EINVAL == %s()\n",wh); 
			exit(1);
		case -EFAULT:
			fprintf(stderr, "Error: EFAULT == %s()\n",wh);
			exit(1);
		case -ENOMEM:
			fprintf(stderr, "Error: ENOMEM == %s()\n",wh);
			exit(1);
		case -ENOSYS:
			fprintf(stderr, "Error: ENOSYS == %s()\n",wh);
			exit(1);
                case -EBADF:
                        fprintf(stderr, "Error: -EBADF == %s()\n",wh);
                        exit(1);
		case -EAGAIN:
			fprintf(stderr, "Error: -EAGAIN == %s()\n",wh);
			exit(1);
		default:
			fprintf(stderr, "Error: unkown(%ld) == %s()\n",result,wh);
			exit(1);
	}
}

void errh_io_submit( long result)
{
	if( result >0 ) return;
	errh_iogeneric("io_submit", result); /* does not return if error */
	if( 0 == result){
		fprintf(stderr, "Error: no events in the event Q (io_submit)\n");
		exit(1);
	}
}


void errh_io_getevents( long result)
{
	if( result >=0 ) return;
	errh_iogeneric("io_getevents", result); 
}

void 	errh_ioerr(signed long long res, signed long long res2, int is_read)
{
	if( res2 == 0 ) return;

	fprintf(stderr, "Error: %lld=%s %s\n",
		res, is_read ? "read()" : "write()", strerror(-res2));
	exit(1);
}

void debug_putpoint(int n)
{
	int i;
	for(i=0; i<n; i++) putchar('.');
	fflush(stdout);
}
void debug_backpoint(int n)
{
	int i;
	for(i=0; i<n; i++) fputs("\b \b",stdout);
	fflush(stdout);
}

/*
 * sub_doio
 * 	generate opt.theradcnt number of concurent io
 *
 */
 
void sub_doio(int is_rand, int readflag)
{
	io_context_t ctx;
	long result;
	struct iocb ** iocbs, **listofiocb;
	struct iocb * myiocbp;
	struct io_event * events;
	long i,needsubmit;
	unsigned long long nextoffset;
	unsigned long long rnd;
	unsigned long long io_qd;
	long iopending;
	int fh;
	struct timeval begint;
	struct timeval endt;
	unsigned long long ofs;


	if(opt.debug){fprintf(stdout,"\n sub_doio start: %s %s \n",
			is_rand?"rnd":"seq",readflag==O_RDONLY?"read":"write"); 
		fflush(stdout);}

	if(is_rand) opt.filesize=getfilesize();

	memset( &ctx, 0, sizeof(ctx));
	result = io_queue_init(opt.threadcnt, &ctx);
	errh_iogeneric("io_queue_init", result );

	iocbs = (struct iocb **) calloc(opt.threadcnt , sizeof(struct iocb *));
	errh_malloc(iocbs, "array1 of struct iocb");
	listofiocb = (struct iocb **) calloc(opt.threadcnt , sizeof(struct iocb *));
	errh_malloc(iocbs, "array2 of struct iocb");
	for( i = 0; i< opt.threadcnt; i++){
		listofiocb[i]=iocbs[i]=(struct iocb *)calloc(1,sizeof(struct iocb));
		errh_malloc(iocbs[i], "one of struct iocb");
	}
	events = calloc(opt.threadcnt , sizeof( struct io_event));
	errh_malloc(events, "array of struct io_events");
	
	fh=open(opt.fname, readflag |  O_LARGEFILE | opt.rawmode );
	if( fh <0 ){
		fprintf(stderr, "Error opening '%s'", opt.fname);
		perror(" ");
		exit(1) ;
	}

	iopending = 0;
	io_qd = 0;
	nextoffset = 0;
	for(i = 0; i < opt.threadcnt; i++){
		if(is_rand){
			read(opt.rndfh, &rnd, sizeof(rnd));
			rnd &= 0x7FFFFFFFFFFFFFFFLL; /*offset is signed */
			nextoffset = opt.mbl * (rnd % (opt.filesize/opt.mbl));
		}
		if(readflag==O_RDONLY){
			io_prep_pread( iocbs[i],fh,opt.buff[i], opt.mbl,nextoffset);
		}else{
			if( opt.uncompressable ) {/* minden 4kiB hatarnal kellene offsetet beirni, 32 biten, ez 16TiB uniq */
				for( ofs= 0 ; ofs<opt.mbl; ofs+=4096 )
					*( (uint32_t *) (opt.buff[i]+ofs) ) = (nextoffset + ofs) >> 12;
			}
			io_prep_pwrite(iocbs[i],fh,opt.buff[i], opt.mbl,nextoffset);
		}
		nextoffset+=opt.mbl;
	}

	gettimeofday(&begint, NULL);
	/* kezdeti */
	result = io_submit( ctx, opt.threadcnt, iocbs);
	errh_io_submit(result);
	io_qd += result;
	iopending += result;
	if(opt.debug==2)printf("\n %12lld %8ld  %ld\n", io_qd, result,iopending);
	if(opt.debug==3)debug_putpoint(iopending);

	do{
		/* varakozas */
		result = io_getevents( ctx, 1, opt.threadcnt, events, NULL);
		errh_io_getevents(result);
		iopending -= result;
		if(opt.debug){
			if(opt.debug==2)
				printf(" %12lld %8ld  %ld\n", io_qd,-result,iopending);
			if(opt.debug==3)debug_backpoint(result);
		}
		/* az a 'result' ami visszatert, a 'opt.totio-io_qd' ami hatravan.
		 * minden viszatertet fel kell dolgozni, de csak a minimumot kell 
		 * ujra elkuldeni.
		*/
		needsubmit=( result< (opt.totio-io_qd)) ? result : opt.totio-io_qd;
		for(i =0; i < result; i++){ /* result == number of terminated events */

			myiocbp=events[i].obj; /* ez a hozza tartozo iocb */
			errh_ioerr(events[i].res,events[i].res2,readflag==O_RDONLY);

			if( i<needsubmit ){
			/* egyedi io */
				if( is_rand){
		                	read(opt.rndfh, &rnd, sizeof(rnd));
		       	        	rnd &= 0x7FFFFFFFFFFFFFFFLL; /* aio_offset is signed */
					nextoffset = opt.mbl * (rnd % (opt.filesize/opt.mbl));
					myiocbp->u.c.offset = nextoffset;
		        	}else{
		                	myiocbp->u.c.offset = nextoffset;
		                	nextoffset += opt.mbl;
		        	}

				if( opt.uncompressable ) {/* minden 4kiB hatarnal kellene offsetet beirni, 32 biten, ez 16TiB uniq */
					for( ofs= 0 ; ofs<opt.mbl; ofs+=4096 )
						*( (uint32_t *) (myiocbp->u.c.buf+ofs) ) = 
							(myiocbp->u.c.offset + ofs) >> 12;
				}
				iocbs[i]=myiocbp; /* iocbs itt ujrahasznalhato? az aio masolja az iocb -t?*/
			}
		}
		if( needsubmit){
			result = io_submit( ctx, needsubmit, iocbs);
			errh_io_submit(result);
			io_qd+=needsubmit;
			iopending+=needsubmit;
			switch(opt.debug){
				case 0: break;
				case 2: printf(" %12lld %8ld  %ld\n",io_qd, 
						needsubmit,iopending);
					break;
				case 3:	debug_putpoint(needsubmit);
					break;
				default:
					if( 0 == ( (io_qd+1) % 65536 )) {
						putchar('+'); 
						fflush(stdout);
					}
					if( 0 == ( ((io_qd+1) * opt.mbl) & 0x3FFFFFFF )) {
						printf(" %lldGiB ", ((io_qd+1)* opt.mbl)>>30);
						fflush(stdout);
					}
			}

		}	

	}while( io_qd < opt.totio );

	result = io_getevents( ctx, iopending, opt.threadcnt, events, NULL);
	errh_io_getevents(result);
	iopending -= result; 
	if(opt.debug==2) printf(" %12lld %8ld  %ld\n", io_qd,-result,iopending);
	
	if( 0 != iopending ){
		fprintf(stderr, "Error: %ld pending ios found (internal failure)\n",iopending);
		exit(1);
	}
	
	fsync(fh); /* az idomeres elott kell lennie, mert van buffer amit ekkor urit*/

	gettimeofday(&endt, NULL);
    {
        double elapsedtime;

        elapsedtime = ((double)endt.tv_sec - (double)begint.tv_sec +
             ((double)endt.tv_usec - (double)begint.tv_usec)/1000000.0);

        printf("elapsed:%f, byteps:%f, iops:%f",
             elapsedtime,
             (double)opt.datasize / elapsedtime,
             (double)opt.totio / elapsedtime); fflush(stdout);
    }
	
	result = io_destroy(ctx);
	errh_iogeneric("io_destroy", result );
	close(fh);
	for( i = 0; i< opt.threadcnt; i++)
		free(listofiocb[i]);
	free(iocbs);
	free(listofiocb);
	if(opt.debug){fprintf(stdout," sub_doio end "); fflush(stdout);}
	
}/* end of sub_doio */


void puthelp(void)
{
	fprintf(stderr,"fillone version %s copyright by Maulis Adam 2024, using AGPL v3 or newer\n\n", VERS); 
	fprintf(stderr,"Usage: fillone [options] filename blocksize datasize\n");
	fprintf(stderr,"   -l lazy: datasize will be rounded up of multiple of blocksize\n");
	fprintf(stderr,"   -p1 sequential write (overwrites data, extends file to datasize) (default test phase)\n");
	fprintf(stderr,"   -p2 random write (overwrites data, does not extend file)\n");
	fprintf(stderr,"   -p3 sequential read\n");
	fprintf(stderr,"   -p4 random read\n");
	fprintf(stderr,"   -r raw: uses O_DIRECT for open (disables local cache)\n");
	fprintf(stderr,"   -u uncompressable and non-deduplicable pattern (the default is the 0xDEADBEEF pattern)\n");
	fprintf(stderr,"   -t# thread count (# means integer)\n\n");
	fprintf(stderr,"Environment variables:\n");
    fprintf(stderr,"    DEBUG  # if not set, there is no debug messages\n"); 
    fprintf(stderr,"    RANDOMPOOL # if not set use %s\n", RANDOMPOOL);
}

int main(int argc, char* argv[])
{
	char * totbuff; /* allocated by malloc for 'opt.buff' */
	int optarg;
	int lazy = 0;
	int isseqwrite=1, isrndwrite=1, isseqread=1,isrndread=1;
	long long i;
	unsigned long long j;
	ssize_t status;
	
	opt.threadcnt = 1;
	opt.rawmode = 0;
	opt.uncompressable = FALSE;

	if( NULL != getenv("DEBUG") ){
		opt.debug = atoi(getenv("DEBUG"));
		if( 0 == opt.debug ) opt.debug = 1;
        printf("Debug messages are enabled.\n");fflush(stdout);
	} else {
		opt.debug = 0;
	}

    if( NULL != getenv("RANDOMPOOL") ){
        opt.randompool = getenv("RANDOMPOOL");
    }else{
        opt.randompool = RANDOMPOOL;
    }
    if(opt.debug){printf(" randompool=%s\n", opt.randompool);fflush(stdout);}

	for(optarg=1;  optarg < argc && argv[optarg][0] == '-'; optarg++ ){
		if(opt.debug){printf(" processing opt%d:%s\n",optarg,argv[optarg]);fflush(stdout);}
		switch( argv[optarg][1] ){
			case 'l': lazy = 1;
				break;
			case 'r': opt.rawmode = O_DIRECT;
				break;
			case 'u': opt.uncompressable = 1;
				break;
			case 'p': if( NULL == strchr(argv[optarg], '1')) isseqwrite = 0;
				  if( NULL == strchr(argv[optarg], '2')) isrndwrite = 0;
				  if( NULL == strchr(argv[optarg], '3')) isseqread = 0;
				  if( NULL == strchr(argv[optarg], '4')) isrndread = 0;
				break;
			case 't':opt.threadcnt = atoi(argv[optarg]+2);
				break;
			default: fprintf(stderr,"Unknown opt: %s\n", argv[optarg]); 
				puthelp();
				return 1;
				break;
		}
	}/*end for optarg */

	if( argc < optarg + 3){
		fprintf(stderr,"Insufficient number of parameters.\n");
		puthelp();
		return 1;
	}
	
	if(1>opt.threadcnt){
		fprintf(stderr,"Threadcount (-t) must be at least 1\n");
		return 1;
	}
	
	opt.fname = strdup(argv[optarg]);
	if(opt.debug){printf(" filename=%s\n",opt.fname);fflush(stdout);}
	optarg++;
	
	opt.mbl = atol( argv[optarg] );
	if( 0 >= opt.mbl){
		fprintf(stderr," Invalid blocksize (%s)\n",argv[optarg]);
		return 1;
	}
	if(opt.debug){printf(" blocksize=%lld\n",opt.mbl);fflush(stdout);}
	optarg++;

	opt.datasize = atoint64(argv[optarg]);
	if( opt.datasize <= 0 ){
		fprintf(stderr," Invalid datasize (%lld != %s)\n",
				opt.datasize, argv[optarg]);
		return 1;
	}
	if(opt.debug){printf(" datasize=%lld\n",opt.datasize);fflush(stdout);}
	
	if( opt.datasize % opt.mbl != 0 ){
		if(lazy){
			if(opt.debug)printf(" datasize was rounded up:%lld ->",
					opt.datasize);
			opt.datasize = (opt.datasize/opt.mbl +1LL) * opt.mbl;
			if(opt.debug){printf("%lld\n", opt.datasize); fflush(stdout);}
		}else{
			fprintf(stderr,"Err: datasize is not multiple of blocksize\n");
			return 1;
		}
	}
	opt.totio = opt.datasize/opt.mbl;
	if(opt.debug){printf(" total ios=%lld\n",opt.totio);fflush(stdout);}
	if( opt.threadcnt > opt.totio){
		opt.threadcnt = opt.totio;
		fprintf(stderr,
			"Warn: to small datasize, threadcount shrink down to %lld\n", 
			opt.threadcnt);
	}

	opt.rndfh = open(opt.randompool, O_RDONLY);
	if( opt.rndfh <0 ){
		fprintf(stderr, "Error opening '%s'", opt.randompool);
		perror(" ");
		exit(1) ;
	}

	opt.buff=(char **)malloc(opt.threadcnt*sizeof(char *) );
	for( i=0 ; i < opt.threadcnt ; i ++ ){
		totbuff = (char *)malloc(roundup4k(opt.mbl)+4096+sizeof(int));
		errh_malloc(totbuff, "one of opt.buff");
		opt.buff[i] = (char *)roundup4k((unsigned long long)totbuff);

		if(opt.debug) printf("totbuff=%p buff[%lld]=%p\n",totbuff,i, opt.buff[i]);

		if( opt.uncompressable ){
			status=read(opt.rndfh, opt.buff[i], roundup4k(opt.mbl) );
			if( 0 == status ) 
				errh_iogeneric("read( randomfile )", errno);
		} else{
			for(j = 0; j<=((roundup4k(opt.mbl)) / sizeof(int)); j++)
	       			((unsigned int *)opt.buff[i])[j]= MAGIC;
		} /* end if uncompressable */
	}

	/*
	 * 		Erdemi munka
	 */


	
	printf("{threadcount:%lld, blocksize:%lld, iocount:%lld, ", opt.threadcnt, opt.mbl, opt.totio);fflush(stdout);

	if(isseqwrite){
        printf("type:\"seqwrite\", ");
        fflush(stdout);
		sub_doio(FALSE, O_WRONLY);
	}else if(isrndwrite){
        printf("type:\"rndwrite\", ");
        fflush(stdout);
		sub_doio(TRUE, O_WRONLY);
	}else if(isseqread){
        printf("type:\"seqread\", ");
        fflush(stdout);
		sub_doio(FALSE, O_RDONLY);
	}else if(isrndread){
        printf("rndread:\"seqread\", ");
        fflush(stdout);
		sub_doio(TRUE, O_RDONLY);
	}
    printf("}\n");
    fflush(stdout);	
    return 0;
}
