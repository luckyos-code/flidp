#!/bin/bash

# Script to trigger all the experiments in seperated slurm jobs

TS=$(date '+%Y-%m-%d_%H:%M:%S');
WORK_DIR="${TS}-all"
