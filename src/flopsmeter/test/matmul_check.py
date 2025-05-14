#!/usr/bin/env python3

import numpy as np
import sys
from time import time

A = np.loadtxt("A.txt")
print("    A.txt loaded, shape:", A.shape)
B = np.loadtxt("B.txt")
print("    B.txt loaded, shape:", B.shape)
C = np.loadtxt("C.txt")
print("    C.txt loaded, shape:", C.shape)

begtim = time()
new_C = A @ B
endtim = time()
print("    matmul time:", endtim-begtim)

print("    Maximum difference between calculated and loaded result:", np.abs(new_C - C).max())

