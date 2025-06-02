.PHONY: clean all test
CCFLAGS = -O4 -Wall -Wextra -march=native -ffast-math -DCL_TARGET_OPENCL_VERSION=300 -DUSE_RESTRICT -DROWPADDING=128
# profiling: -pg
CC = gcc
LINKFLAGS = -lOpenCL -lpthread -lquadmath -lm
LINKER = gcc
# profiling: -pg

ALL = flopsmeter opencl_matmul_test matmul_test gener_tiling_test
TESTFILES = A.txt B.txt C.txt

all: $(ALL)

flopsmeter: flopsmeter.o matmul.o
	$(LINKER) -o flopsmeter flopsmeter.o matmul.o $(LINKFLAGS)

flopsmeter.o: flopsmeter.c matmul.h
	$(CC) -c -o flopsmeter.o flopsmeter.c $(CCFLAGS)

matmul.o: matmul.h matmul.c matmul.inc
	$(CC) -c -o matmul.o matmul.c $(CCFLAGS)

matmul_test: matmul_test.o matmul.o
	$(LINKER) -o matmul_test matmul_test.o matmul.o $(LINKFLAGS)

matmul_test.o: matmul_test.c matmul.h
	$(CC) -c -o matmul_test.o matmul_test.c $(CCFLAGS)

opencl_matmul_test: opencl_matmul_test.o opencl_matmul.o opencl_helper.o matmul.o gener_tiling.o
	$(LINKER) -o opencl_matmul_test opencl_matmul_test.o opencl_matmul.o opencl_helper.o matmul.o gener_tiling.o $(LINKFLAGS)

opencl_matmul_test.o: opencl_matmul_test.c opencl_helper.h
	$(CC) -c -o opencl_matmul_test.o opencl_matmul_test.c $(CCFLAGS) 

opencl_matmul.o: opencl_matmul.c opencl_matmul.inc opencl_matmul.h opencl_matmul.cl opencl_matmul.cl.inc
	$(CC) -c -o opencl_matmul.o opencl_matmul.c $(CCFLAGS)

opencl_helper.o: opencl_helper.c opencl_helper.h
	$(CC) -c -o opencl_helper.o opencl_helper.c $(CCFLAGS)

gener_tiling.o: gener_tiling.c gener_tiling.h
	$(CC) -c -o gener_tiling.o gener_tiling.c $(CCFLAGS)

gener_tiling_test.o: gener_tiling_test.c
	$(CC) -c -o gener_tiling_test.o gener_tiling_test.c $(CCFLAGS)

gener_tiling_test: gener_tiling.o gener_tiling_test.o
	$(LINKER) -o gener_tiling_test gener_tiling_test.o gener_tiling.o $(LINKFLAGS)


test: A.txt B.txt opencl_matmul_test matmul_test test/matmul_check.py
	./matmul_test fp32 A.txt B.txt C.txt
	test/matmul_check.py A.txt B.txt C.txt
	./matmul_test fp64 A.txt B.txt C.txt
	test/matmul_check.py A.txt B.txt C.txt
	./matmul_test fp80 A.txt B.txt C.txt
	test/matmul_check.py A.txt B.txt C.txt
	./matmul_test fp128 A.txt B.txt C.txt
	test/matmul_check.py A.txt B.txt C.txt

A.txt: test/matmul_gener.py
	test/matmul_gener.py 1511 2003 2999

clean:
	rm -f *.o $(ALL) $(TESTFILES) gmon.out
