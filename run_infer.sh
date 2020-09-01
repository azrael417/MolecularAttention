#!/bin/bash

# unimproved run
python infer_opt.py \
       -i /data/molecular_attention/smiles.smi \
       -trt /data/molecular_attention/model.trt \
       -o /data/runs/attention/ref.csv \
       -m /data/molecular_attention/model.pt \
       -i /data/molecular_attention/smiles.smi \
       -o /data/runs/attention/ref.csv \
       -b 256 -j 32 -dtype=fp16 \
       -num_calibration_batches=10

#python infer_q.py \
#       -i /data/molecular_attention/smiles.smi \
#       -q /data/molecular_attention/model.ptq \
#       -o /data/runs/attention/ref.csv \
#       -m /data/molecular_attention/model.pt \
#       -i /data/molecular_attention/smiles.smi \
#       -o /data/runs/attention/ref.csv \
#       -b 256 -j 32
