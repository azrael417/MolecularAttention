import os
import sys
import glob
import time
import numpy as np
from argparse import ArgumentParser
from progressbar import ProgressBar, Bar, Counter, AdaptiveTransferSpeed

# torch stuff
import torch
from torch.utils.data import DataLoader

# MPI
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# add the parent path too
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# custom stuff
from datasets import CompressedMoleculesDataset, shard_init_fn
from models import imagemodel


def main(args_i):

    if args_i.distributed:
        # MPI wireup
        MPI.Init_thread()
    
        # get communicator: duplicate from comm world
        comm = MPI.COMM_WORLD.Dup()
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()
        have_mpi = True
    else:
        print("MPI not found or supported, running single process")
        comm_size = 1
        comm_rank = 0
        have_mpi = False

    # set device
    device = torch.device(f"cuda:{args_i.d}")
    torch.backends.cudnn.benchmark = True

    # get files and shard list
    filelist = sorted(glob.glob(args_i.i))
    totalsize = len(filelist)
    shardsize = int(np.ceil(totalsize / comm_size))
    start = comm_rank * shardsize
    end = min([start + shardsize, totalsize])
    if comm_rank == 0:
        print(f"Found {totalsize} files. Deploying about {shardsize} per rank.")

    # get data loader
    dset = CompressedMoleculesDataset(filelist, start, end, encoding = "smiles", max_prefetch_count = 3)

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

    # set up a buffer for the accumulation of samples
    samples = np.zeros(1, dtype = np.int64)
    inc_buffer = np.zeros(1, dtype = np.int64)

    if have_mpi:
        samples_win = MPI.Win.Create(samples, comm = comm)

    # do the inference step
    if comm_rank == 0:
        print("Starting Inference")
        widgets = ['Running: ', Counter(), ' ', Bar(marker='#',left='[',right=']'), ' ', AdaptiveTransferSpeed(), ' samples/s']
        pbar = ProgressBar(widgets = widgets)
        pbar.start()

    if have_mpi:
        comm.barrier()
    samples = 0
    duration = time.time()
    with torch.no_grad():
        results = []
        for idx, (im, identifier, fname) in enumerate(train_loader):
                
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
            inc_buffer += pred.shape[0]

            print(comm_rank, ' ', inc_buffer)

            # update counter and report if requested
            if (idx % args_i.update_frequency == 0):
                if have_mpi:
                    samples_win.Lock(0, MPI.LOCK_SHARED)
                    samples_win.Accumulate(inc_buffer, 0, op=MPI.SUM)
                    samples_win.Unlock(0)
                else:
                    samples += inc_buffer
                inc_buffer = 0

                if (comm_rank == 0):
                    if have_mpi:
                        #lock window
                        samples_win.Lock(0, MPI.LOCK_EXCLUSIVE)

                    #update pbar
                    pbar.update(np.asscalar(samples))

                    if have_mpi:
                        #unlock window
                        samples_win.Unlock(0)
        
        # final update
        if have_mpi:
            samples_win.Lock(0, MPI.LOCK_SHARED)
            samples_win.Accumulate(inc_buffer, 0, op=MPI.SUM)
            samples_win.Unlock(0)
        else:
            samples += inc_buffer
        inc_buffer = 0

    # sync
    if have_mpi:
        comm.barrier()

    # timer
    duration = time.time() - duration

    # get samples
    samples_count = np.asscalar(samples)
    
    if comm_rank == 0:
        print(f"Processed {samples_count} samples in {duration}s, throughput {samples_count/duration} samples/s")
                

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', type=str, required=True)
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-trt', type=str, required=False, default=None)
    parser.add_argument('-dtype', type=str, choices=["fp32", "fp16", "int8"], required=False)
    parser.add_argument('-num_calibration_batches', type=int, default=1000, required=False)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--update_frequency', type=int, required=False, default=100)
    parser.add_argument('-d', type=int, required=False, default=0)
    parser.add_argument('-j', type=int, required=False, default=1)
    parser.add_argument('-b', type=int, required=False, default=64)
    pargs =  parser.parse_args()

    main(pargs)
