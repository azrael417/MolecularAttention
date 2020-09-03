import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from features import datasets
from train import load_data_models
from models import imagemodel


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', type=str, required=True)
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-q', type=str, required=False, default=None)
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

smiles_file = ""
trues = pd.read_csv(args_i.i)
trues = trues.iloc[:, 0].tolist()

model = imagemodel.ImageModel(nheads=args.nheads, outs=args.t, classifacation=args.classifacation, dr=0.,
                            intermediate_rep=args.width, linear_layers=args.depth, pretrain=False,
                            quantize=True)
model.load_state_dict(torch.load(file, map_location='cpu')['model_state'])
model.eval()

# convert to trt if requested
if args_i.q is not None:
    import torch.quantization
    
    # set quantization config for server (x86)
    model.qconfig = torch.quantization.default_qconfig

    # insert observers
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate the model and collect statistics

    # convert to quantized version
    torch.quantization.convert(model, inplace=True)

smiles_file = ""
trues = pd.read_csv(args_i.i)
trues = trues.iloc[:, 0].tolist()

dset = datasets.ImageDatasetInfer(trues)
train_loader = DataLoader(dset, num_workers = args_i.j,
                          pin_memory = True,
                          batch_size = args_i.b,
                          shuffle = False)

with torch.no_grad():
    with open(args_i.o, 'w') as f:
        for i, (im, smile) in tqdm(enumerate(train_loader)):

            # forward pass
            pred, _ = model(im)
            
            #pred = pred.cpu().numpy().flatten()
            #for j, smi in enumerate(smile):
            #    f.write(smi + "," + str(pred[j]) + "\n")
