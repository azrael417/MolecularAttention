#!/bin/bash

# devid
device=${OMPI_COMM_WORLD_LOCAL_RANK:=0}

# data root
#data_root="/gpfs/alpine/proj-shared/med110/cov19_data"
data_root="/gpfs/alpine/med110/world-shared/ULT911"

# unimproved run
python -u infer_images.py \
    -t 500 \
    -d ${device} \
    -i "${data_root}/images_compressed/*.pkl.gz" \
    -trt /gpfs/alpine/proj-shared/med110/tkurth/attention_meta/model_6W02.trt \
    -o /gpfs/alpine/proj-shared/med110/cov19_data/scores \
    -m /gpfs/alpine/proj-shared/med110/tkurth/attention_meta/model_6W02.pt \
    --output_frequency 200 \
    -b 256 -j 12 -dtype=fp16 \
    -num_calibration_batches=10 ${1}

#python infer_q.py \
#       -i /data/molecular_attention/smiles.smi \
#       -q /data/molecular_attention/model.ptq \
#       -o /data/runs/attention/ref.csv \
#       -m /data/molecular_attention/model.pt \
#       -i /data/molecular_attention/smiles.smi \
#       -o /data/runs/attention/ref.csv \
#       -b 256 -j 32
