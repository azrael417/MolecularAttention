import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from features import datasets
from train import load_data_models

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', type=str, required=True)
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-j', type=int, required=False, default=0)
    parser.add_argument('-b', type=int, required=False, default=256)
    return parser.parse_args()

if torch.cuda.is_available():
    import torch.backends.cudnn
    torch.backends.cudnn.benchmark = True

args_i = get_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file = args_i.m
args = torch.load(file, map_location=torch.device('cpu'))['args']

smiles_file = ""
trues = pd.read_csv(args_i.i)
trues = trues.iloc[:, 0].tolist()

values = np.zeros(len(trues))
# dset, _, model = load_data_models("moses/test_scaffolds.smi", 32, 1, 1, 'weight', nheads=8, dropout=0.1, return_datasets=True, precompute_frame="moses/test_scaffolds_weight.npy", intermediate_rep=128)
_, _, model = load_data_models(args_i.i, args.r, args.w, args.b, args.p,
                                  nheads=args.nheads,
                                  precompute_frame=values,
                                  imputer_pickle=args.imputer_pickle,
                                  tasks=args.t, rotate=0,
                                  classifacation=args.classifacation, ensembl=args.ensemble_eval,
                                  dropout=args.dropout_rate,
                                  intermediate_rep=args.width, depth=args.depth,
                                  bw=args.bw, return_datasets=True)
model.load_state_dict(torch.load(file, map_location='cpu')['model_state'])
model = model.to(device)
model.eval()

smiles_file = ""
trues = pd.read_csv(args_i.i)
trues = trues.iloc[:, 0].tolist()

dset = datasets.ImageDatasetInfer(trues)
train_loader = DataLoader(dset, num_workers=args_i.j, pin_memory=True, batch_size=args_i.b,
                                  shuffle=False)

with torch.no_grad():
    with open(args_i.o, 'w') as f:
        for i, (im, smile) in tqdm(enumerate(train_loader)):
            im = im.to(device)
            pred, _ = model(im)
            pred = pred.cpu().numpy().flatten()
            for smi in smile:
                f.write(smi + "," + str(pred[i]) + "\n")
