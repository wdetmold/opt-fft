#!/bin/sh
# Regenerate implementation.c exactly as the graded attempt did (deterministic).
# Requires python3 + mpmath. Run from this directory.
KB=2 python3 codegen.py
