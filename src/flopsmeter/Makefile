.PHONY: clean
flopsmeter: flopsmeter.c
	gcc -O4 -o flopsmeter  flopsmeter.c -march=native -DUSE_RESTRICT -Wall  -ffast-math -lpthread
clean:
	rm -f flopsmeter
