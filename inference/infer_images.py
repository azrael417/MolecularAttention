import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser

# add the parent path too
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# custom stuff
from datasets import CompressedImageDataset, shard_init_fn
from models import imagemodel


def main(args_i):

    # set device
    device = torch.device(f"cuda:{args_i.d}")
    torch.backends.cudnn.benchmark = True

    # get data loader
    dset = CompressedImageDataset(args_i.i)
    train_loader = DataLoader(dset, num_workers = args_i.j,
                              pin_memory = True,
                              batch_size = args_i.b,
                              shuffle = False,
                              worker_init_fn = shard_init_fn)
    
    # set model
    modelfile = args_i.m
    args = torch.load(modelfile, map_location=torch.device('cpu'))['args']
    if args.width is None:
        args.width = 256
    model = imagemodel.ImageModel(nheads=args.nheads, outs=args.t, classifacation=args.classifacation, dr=0.,
                                  intermediate_rep=args.width, linear_layers=args.depth, pretrain=False)
    model.load_state_dict(torch.load(file, map_location='cpu')['model_state'])
    model.eval()
    
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

        # generate trt model
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

    exit

    with torch.no_grad():
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
                

if __name__ == "__main__":
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
    pargs =  parser.parse_args()

    main(pargs)
