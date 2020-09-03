import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# add the parent path too
from pathlib import Path
sys.path.append(Path('.').parent)

# custom stuff
from features import datasets
from models import imagemodel

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', type=str, required=True)
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-trt', type=str, required=False, default=None)
    parser.add_argument('-dtype', type=str, choices=["fp32", "fp16", "int8"], required=False)
    parser.add_argument('-num_calibration_batches', type=int, default=1000, required=False)
    parser.add_argument('-d', type=int, required=False, default=0)
    parser.add_argument('-j', type=int, required=False, default=1)
    parser.add_argument('-b', type=int, required=False, default=64)
    return parser.parse_args()

if torch.cuda.is_available():
    import torch.backends.cudnn
    torch.backends.cudnn.benchmark = True

args_i = get_args()

device = torch.device(f"cuda:{args_i.d}" if torch.cuda.is_available() else "cpu")

file = args_i.m
args = torch.load(file, map_location=torch.device('cpu'))['args']
if args.width is None:
    args.width = 256

smiles_file = ""
trues = pd.read_csv(args_i.i)
trues = trues.iloc[:, 0].tolist()

values = np.zeros(len(trues))
# dset, _, model = load_data_models("moses/test_scaffolds.smi", 32, 1, 1, 'weight', nheads=8, dropout=0.1, return_datasets=True, precompute_frame="moses/test_scaffolds_weight.npy", intermediate_rep=128)
#_, _, model = load_data_models(args_i.i, args.r, args.w, args.b, args.p,
#                                  nheads=args.nheads,
#                                  precompute_frame=values,
#                                  imputer_pickle=args.imputer_pickle,
#                                  tasks=args.t, rotate=0,
#                                  classifacation=args.classifacation, ensembl=args.ensemble_eval,
#                                  dropout=0.,
#                                  intermediate_rep=args.width, depth=args.depth,
#                                  bw=args.bw, return_datasets=True)

model = imagemodel.ImageModel(nheads=args.nheads, outs=args.t, classifacation=args.classifacation, dr=0.,
                              intermediate_rep=args.width, linear_layers=args.depth, pretrain=False)
model.load_state_dict(torch.load(file, map_location='cpu')['model_state'])
model = model.to(device)
model.eval()

# dataset
smiles_file = ""
trues = pd.read_csv(args_i.i)
trues = trues.iloc[:, 0].tolist()

dset = datasets.ImageDatasetInfer(trues)
train_loader = DataLoader(dset, num_workers = args_i.j,
                          pin_memory = True,
                          batch_size = args_i.b,
                          shuffle = False)

# convert to half if requested
if args_i.dtype == "fp16":
    model = model.half()

# convert to trt if requested
if args_i.trt is not None:
    import tensorrt as trt
    from torch2trt import torch2trt
    # dummy input
    inp = torch.ones((1, 3, 128, 128)).to(device)
    if args_i.dtype == "fp16":
        inp = inp.half()
    
    cal_dset = torch.utils.data.Subset(dset, list(range(0, args_i.b * args_i.num_calibration_batches)))
    
    model_trt = torch2trt(model, [inp], 
                        log_level=trt.Logger.INFO, 
                        max_workspace_size=1<<26, 
                        fp16_mode=True if args_i.dtype == "fp16" else False,
                        int8_mode=True if args_i.dtype == "int8" else False,
                        int8_calib_dataset = cal_dset if args_i.dtype == "int8" else None,
                        max_batch_size=args_i.b)

    # save trt model
    torch.save(model_trt.state_dict(), args_i.trt)


# pick the FW pass model
model_fw = model_trt if args_i.trt is not None else model

with torch.no_grad():
    with open(args_i.o, 'w') as f:
        with tqdm(total=len(train_loader), unit='samples') as pbar:
            for i, (im, smile) in enumerate(train_loader):
                
                # convert to half if requested
                if args_i.dtype == "fp16":
                    im = im.half()
                elif args_i.dtype == "int8":
                    im = im.to(torch.int8)

                # upload data
                im = im.to(device)
                
                # forward pass
                pred = model_fw(im)

                # update progress
                pbar.update(args_i.b)
                
                #pred = pred.cpu().numpy().flatten()
                #for j, smi in enumerate(smile):
                #    f.write(smi + "," + str(pred[j]) + "\n")
