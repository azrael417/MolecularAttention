#!/bin/bash

# unimproved run
python infer_images.py \
       -i "/gpfs/alpine/proj-shared/med110/cov19_data/*/*.pkl.gz" \
       -trt /gpfs/alpine/proj-shared/med110/tkurth/attention_meta/model.trt \
       -o /gpfs/alpine/proj-shared/med110/cov19_data/scores \
       -m /gpfs/alpine/proj-shared/med110/tkurth/attention_meta/model.pt \
       -b 256 -j 8 -dtype=fp16 \
       -num_calibration_batches=10

#python infer_q.py \
#       -i /data/molecular_attention/smiles.smi \
#       -q /data/molecular_attention/model.ptq \
#       -o /data/runs/attention/ref.csv \
#       -m /data/molecular_attention/model.pt \
#       -i /data/molecular_attention/smiles.smi \
#       -o /data/runs/attention/ref.csv \
#       -b 256 -j 32
