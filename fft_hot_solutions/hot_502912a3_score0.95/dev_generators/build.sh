#!/bin/bash
set -e
cd /tmp/dev/gen
python3 assemble.py
rm -f implementation.so
gcc -O3 -march=native -funroll-loops -shared -fPIC implementation.c -o implementation.so -lm
test -f implementation.so && echo BUILD-OK
