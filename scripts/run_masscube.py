#!/usr/bin/env python3
"""Run MassCube's default untargeted workflow on a prepared project directory.

The only change from a plain `untargeted-metabolomics` call: multiprocessing.cpu_count()
is pointed at the CPUs Slurm actually gave the job. MassCube sizes its worker pool as
0.8 x cpu_count(), which on a shared node is the whole machine (64-224 cores), not the
allocation. No detection parameter is touched.
usage: run_masscube.py PROJECT_DIR
"""
import multiprocessing
import os
import sys

ncpu = len(os.sched_getaffinity(0))
multiprocessing.cpu_count = lambda: ncpu
os.cpu_count = lambda: ncpu

from masscube.workflows import untargeted_metabolomics_workflow  # noqa: E402

untargeted_metabolomics_workflow(path=sys.argv[1])
