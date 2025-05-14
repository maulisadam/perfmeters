#!/usr/bin/env python3

import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: ./matmul_gener.py 2000 [2500 3000]")
    print("    where 2000 is a row and column size of the generated matrix.")
    print("    Or M1 will be 2000x2500, M2 will be 2500x3000")
    print("    Generated matrixes written to A.txt and B.txt")
    sys.exit(1)

X = int(sys.argv[1])
if len(sys.argv) > 2:
    Y = int(sys.argv[2])
    Z = int(sys.argv[3])
else:
    Y = X
    Z = X

np.savetxt("A.txt", np.random.uniform(size=(X,Y)))
np.savetxt("B.txt", np.random.uniform(size=(Y,Z)))
