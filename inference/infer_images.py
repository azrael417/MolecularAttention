import os
import sys
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from argparse import ArgumentParser

# MPI
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# add the parent path too
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# custom stuff
from datasets import CompressedImageDataset, shard_init_fn
from models import imagemodel


def main(args_i):

    # MPI wireup
    MPI.Init_thread()
    
    # get communicator: duplicate from comm world
    comm = MPI.COMM_WORLD.Dup()
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # set device
    device = torch.device(f"cuda:{args_i.d}")
    torch.backends.cudnn.benchmark = True

    # get data loader
    dset = CompressedImageDataset(args_i.i, max_prefetch_count = 3)

    # shard dataset
    totalsize = len(dset)
    shardsize = int(np.ceil(totalsize / comm_size))
    dset.start = comm_rank * shardsize
    dset.end = min([dset.start + shardsize, totalsize])

    # create train loader
    train_loader = DataLoader(dset, num_workers = args_i.j,
                              pin_memory = True,
                              batch_size = args_i.b,
                              shuffle = False,
                              drop_last = False,
                              worker_init_fn = shard_init_fn)
    
    # set model
    modelfile = args_i.m
    args = torch.load(modelfile, map_location=torch.device('cpu'))['args']

    if args.width is None:
        args.width = 256
    model = imagemodel.ImageModel(nheads=args.nheads, outs=args.t, classifacation=args.classifacation, dr=0.,
                                  intermediate_rep=args.width, linear_layers=args.depth, pretrain=False)
    model.load_state_dict(torch.load(modelfile, map_location='cpu')['model_state'])
    model = model.to(device)
    model.eval()
    
    # convert to half if requested
    if args_i.dtype == "fp16":
        model = model.half()

    # convert to trt if requested
    if args_i.trt is not None:
        import tensorrt as trt
        from torch2trt import torch2trt, TRTModule

        # check if model exists:
        if os.path.isfile(args_i.trt):
            if comm_rank == 0:
                print("Loading TRT model")
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(args_i.trt))

        else:
            if comm_rank == 0:
                print("Generating TRT model")

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

    # do the inference step
    if comm_rank == 0:
        print("Starting Inference")
    comm.barrier()
    samples = 0
    duration = time.time()
    with torch.no_grad():
        for i, (im, identifier) in enumerate(train_loader):
                
                # convert to half if requested
                if args_i.dtype == "fp16":
                    im = im.half()
                elif args_i.dtype == "int8":
                    im = im.to(torch.int8)

                # upload data
                im = im.to(device)
                
                # forward pass
                pred = model_fw(im)

                # sample counter
                samples += pred.shape[0]
    
    
    # sync
    comm.barrier()

    # timer
    duration = time.time() - duration

    # update counter
    samples_arr = np.asarray(samples, dtype = np.int64)
    samples = np.asscalar(comm.allreduce(samples_arr, op=MPI.SUM))
    
    if comm_rank == 0:
        print(f"Processed {samples} samples in {duration}s, throughput {samples/duration} samples/s")
                

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
