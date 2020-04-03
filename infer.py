import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from models.imagemodel import ImageModel
from train import load_data_models
from features import datasets
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as TT
from torch.utils.data import DataLoader
from features.utils import Invert

if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file = 'saved_models/adpr.pt'
args = torch.load(file, map_location=torch.device('cpu'))['args']

# dset, _, model = load_data_models("moses/test_scaffolds.smi", 32, 1, 1, 'weight', nheads=8, dropout=0.1, return_datasets=True, precompute_frame="moses/test_scaffolds_weight.npy", intermediate_rep=128)
_, _, model = load_data_models("adrp_adpr_pocket1/smiles.smi", args.r, args.w, args.b, args.p,
                                  nheads=args.nheads,
                                  precompute_frame="adrp_adpr_pocket1/adrp_adpr_pocket1_values3.npy",
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
trues = pd.read_csv("ADRP-ADPR_pocket1_receptor.out",
                   header=None, names=['name', 'smiles', 'dock_true', 'U', 'dset','pl'], low_memory=False)
trues = trues.smiles.tolist()

dset = datasets.ImageDatasetInfer(trues)
train_loader = DataLoader(dset, num_workers=48, pin_memory=True, batch_size=256,
                                  shuffle=False)

with torch.no_grad():
    with open("out_infer.txt", 'w') as f:
        for i, (im, smile) in tqdm(enumerate(train_loader)):
            im = im.to(device)
            pred, _ = model(im)
            pred = pred.cpu().numpy().flatten()
            for smi in smile:
                f.write(smi + "," + str(pred[i]) + "\n")
