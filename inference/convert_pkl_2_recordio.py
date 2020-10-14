import os
import h5py as h5
import numpy as np
import mxnet as mx
from tqdm import tqdm

# data paths
input_dir = "/global/cscratch1/sd/tkurth/rbc_data/raw"
output_dir = "/global/cscratch1/sd/tkurth/rbc_data/mxnet_transpose"
data_format = "nhwc"

# get filoes
files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".h5")]

# create output dirs
if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

for fname in files:
    # get basename
    ofname = os.path.splitext(os.path.join(output_dir, os.path.basename(fname)))[0]

    # open hdf5 file
    with h5.File(fname, 'r') as f:
        # HR
        if not os.path.isfile(ofname + '_combined.idx'):
            nrecs = f["fields_hr"].shape[0]
            assert(nrecs == f["fields_tilde"].shape[0])
            record = mx.recordio.MXIndexedRecordIO(ofname + '_combined.idx', ofname + '_combined.rec', 'w')
            for idx in tqdm(range(nrecs)):
                # header
                header = mx.recordio.IRHeader(0, 0., idx, 0)
                tilde_arr = f["fields_tilde"][idx, ...]
                hr_arr = f["fields_hr"][idx, ...]

                if data_format == "nhwc":
                    tilde_arr = np.transpose(tilde_arr, axes = [2,1,0])
                    hr_arr = np.transpose(hr_arr, axes = [2,1,0])
                
                token = mx.recordio.pack(header, tilde_arr.tobytes() + hr_arr.tobytes())
                record.write_idx(idx, token)
            # close
            record.close()
        else:
            print("Record {} already exists, skipping.".format(ofname))
            continue
