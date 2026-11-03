#!/bin/bash
# one short job: the estimator on one real injection, read-only
cd /quobyte/metabolomicsgrp/fanzhou/hoffmann || { echo NO_MOUNT $(hostname); exit 9; }
export NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/p3d-numba-$$"
exec envs/peak3d/bin/python presentation/param_estimation/extract.py
