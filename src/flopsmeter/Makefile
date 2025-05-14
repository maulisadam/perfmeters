.PHONY: clean all test
CCFLAGS = -O4 -Wall -Wextra -march=native -ffast-math
CC = gcc
ALL = flopsmeter opencl_matmul_test matmul_test
TESTFILES = A.txt B.txt C.txt

all: $(ALL)

flopsmeter: flopsmeter.o matmul.o
	$(CC) -o flopsmeter flopsmeter.o matmul.o -lpthread -lquadmath

flopsmeter.o: flopsmeter.c matmul.h
	$(CC) -c -o flopsmeter.o flopsmeter.c $(CCFLAGS) -DUSE_RESTRICT

matmul.o: matmul.h matmul.c matmul.inc
	$(CC) -c -o matmul.o matmul.c $(CCFLAGS)

matmul_test: matmul_test.o matmul.o
	$(CC) -o matmul_test matmul_test.o matmul.o -lpthread -lquadmath

matmul_test.o: matmul_test.c matmul.h
	$(CC) -c -o matmul_test.o matmul_test.c $(CCFLAGS) -DUSE_RESTRICT

opencl_matmul_test: opencl_matmul_test.o opencl_helper.o
	$(CC) -o opencl_matmul_test opencl_matmul_test.o opencl_helper.o -lOpenCL

opencl_matmul_test.o: opencl_matmul_test.c opencl_helper.h
	$(CC) -c -o opencl_matmul_test.o opencl_matmul_test.c  $(CCFLAGS) -DCL_TARGET_OPENCL_VERSION=220

opencl_helper.o: opencl_helper.c opencl_helper.h
	$(CC) -c -o opencl_helper.o opencl_helper.c $(CCFLAGS) -DCL_TARGET_OPENCL_VERSION=220

test: A.txt B.txt opencl_matmul_test matmul_test test/matmul_check.py
	./matmul_test fp32 A.txt B.txt C.txt
	test/matmul_check.py A.txt B.txt C.txt
	./matmul_test fp64 A.txt B.txt C.txt
	test/matmul_check.py A.txt B.txt C.txt

A.txt: test/matmul_gener.py
	test/matmul_gener.py 1511 2003 2999


clean:
	rm -f *.o $(ALL) $(TESTFILES)
