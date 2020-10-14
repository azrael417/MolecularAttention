#!/bin/bash
#BSUB -W 120
#BSUB -P VEN201
#BSUB -J eval
#BSUB -alloc_flags "NVME"

# set env
source ../../python_env/env.sh

# activate conda
source activate /ccs/home/tkurth/project/attention/python_env/conda_attention_cuda-110_py-36

# we need that here too
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)

# run the stuff
jsrun -n ${nnodes} -r 1 -g 6 -a 6 -c 42 -d packed ./run_infer.sh --distributed
