#!/bin/bash

# stagein
stagein=1

# devid
device=${OMPI_COMM_WORLD_LOCAL_RANK:=0}

data_root="/gpfs/alpine/proj-shared/med110/cov19_data"
if [ "${stagein}" == "1" ]; then
    if [ "${device}" == "0" ]; then
	mkdir -p /tmp/tkurth/cov19_data/
	rsync -rva ${data_root}/* /tmp/tkurth/cov19_data/
    fi
    data_root="/tmp/tkurth/cov19_data"
fi

# unimproved run
python -u infer_images.py \
    -d ${device} \
    -i "${data_root}/*/*.pkl.gz" \
    -trt /gpfs/alpine/proj-shared/med110/tkurth/attention_meta/model.trt \
    -o /gpfs/alpine/proj-shared/med110/cov19_data/scores \
    -m /gpfs/alpine/proj-shared/med110/tkurth/attention_meta/model.pt \
    -b 256 -j 8 -dtype=fp16 \
    -num_calibration_batches=10 ${1}

#python infer_q.py \
#       -i /data/molecular_attention/smiles.smi \
#       -q /data/molecular_attention/model.ptq \
#       -o /data/runs/attention/ref.csv \
#       -m /data/molecular_attention/model.pt \
#       -i /data/molecular_attention/smiles.smi \
#       -o /data/runs/attention/ref.csv \
#       -b 256 -j 32
