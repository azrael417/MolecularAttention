import os
import re
import sys
import glob
import time
import tempfile
import itertools
import numpy as np
from argparse import ArgumentParser
from progressbar import ProgressBar, Bar, Counter, Timer, AdaptiveTransferSpeed, UnknownLength
import pandas as pd
import concurrent.futures as cf
from shutil import copyfile

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

def stage_subset(target_directory, filelist, max_retry):
    if os.path.isdir(target_directory):
        newfilelist = []
        for finame in filelist:
            # construct output path:
            bname = os.path.basename(finame)
            foname = os.path.join(target_directory, bname)
            retry = 0
            while (retry < max_retry):
                try:
                    copyfile(finame, foname)
                    newfilelist.append(foname)
                    break
                except IOError as e:
                    retry += 1

        return newfilelist
    else:
        return filelist

def stage_files(stage_directory, filelist, max_retry = 5, num_workers = 1, executor = None):
    handles = []
    if os.path.isdir(stage_directory):
        # divide into subsets:
        fullsize = len(filelist)
        chunksize = int(np.ceil(fullsize / num_workers))
        for i in range(num_workers):
            start = chunksize * i
            end = min([start + chunksize, fullsize])
            sublist = filelist[start:end]
            handles.append(executor.submit(stage_subset, stage_directory, sublist, max_retry))
        # create new list with file locations
        newlist = [os.path.join(stage_directory, os.path.basename(f)) for f in filelist]
    else:
        newlist = filelist

    return handles, newlist


def write_ligand_files(filename, df, file_per_ligand = False):
    # save to csv
    df.to_csv(filename)

    print(f"Wrote ligands to {filename}.")

def main(args_i):

    # timer
    setup_duration = time.time()

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

    if comm_rank == 0:
        # create output dir if not exists
        if not os.path.isdir(args_i.o):
            os.makedirs(args_i.o, exist_ok = True)
    if have_mpi:
        comm.barrier()

    # get files and shard list
    filelist = sorted(glob.glob(args_i.i))
    totalsize = len(filelist)
    shardsize = int(np.ceil(totalsize / comm_size))
    start = comm_rank * shardsize
    end = min([start + shardsize, totalsize])
    filelist = filelist[start:end]
    if comm_rank == 0:
        print(f"Found {totalsize} files. Deploying about {shardsize} per rank.")

    #debug
    #filelist = filelist[:10]
    #debug
    stage_handles = []
    if args_i.stage_dir is not None:
        if comm_rank == 0:
            print("Starting file staging")
        tmpdir = tempfile.TemporaryDirectory(prefix="inference", dir=args_i.stage_dir)
        thread_executor = cf.ThreadPoolExecutor(max_workers = args_i.num_stage_workers)
        stage_handles, filelist = stage_files(tmpdir.name, filelist, args_i.num_stage_workers, executor = thread_executor)
    
    # set model
    if comm_rank == 0:
        print("Starting loading model")
    modelfile = args_i.m
    match = re.match(r"^model_(.*?).pt$", os.path.basename(args_i.m))
    receptor_id = "N/A" if match is None else match.groups()[0]
    args = torch.load(modelfile, map_location=torch.device('cpu'))['args']

    if args.width is None:
        args.width = 256
    model = imagemodel.ImageModel(nheads=args.nheads, outs=args.t, classifacation=args.classifacation, dr=0.,
                                  intermediate_rep=args.width, linear_layers=args.depth, pretrain=False)
    model.load_state_dict(torch.load(modelfile, map_location='cpu')['model_state'])
    model = model.to(device)
    model.eval()
    if comm_rank == 0:
        print("Finished loading model")
    
    # convert to half if requested
    if args_i.dtype == "fp16":
        model = model.half()
        
    # dataset here
    dset = CompressedMoleculesDataset(filelist, start = 0, end = len(filelist), \
                                      encoding = "images", max_prefetch_count = 3)

    # create train loader
    infer_loader = DataLoader(dset, num_workers = args_i.j,
                              pin_memory = True,
                              batch_size = args_i.b,
                              shuffle = False,
                              drop_last = False,
                              worker_init_fn = shard_init_fn)

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
            if comm_rank == 0:
                torch.save(model_trt.state_dict(), args_i.trt)

    # pick the FW pass model
    model_fw = model_trt if args_i.trt is not None else model

    # set up a buffer for the accumulation of samples
    samples_buffer = np.zeros([1], dtype = np.int64)
    inc_buffer = np.zeros([1], dtype = np.int64)

    if have_mpi:
        if comm_rank == 0:
            samples_win = MPI.Win.Create(samples_buffer, comm = comm)
        else:
            samples_win = MPI.Win.Create(None, comm = comm)

    # make sure data was staged:
    if stage_handles:
        stage_duration = time.time()
        flist = [sh.result() for sh in stage_handles]
        filelist = list(itertools.chain.from_iterable(flist))
        stage_duration = time.time() - stage_duration
        infer_loader.dataset.update_filelist(filelist, 0, len(filelist))

    # some last things we want to setup
    if comm_rank == 0:
        print("Starting Inference")
        # set up progress bar
        widgets = ['Running: ', Counter(), ' ', \
                   AdaptiveTransferSpeed(prefix='', unit='samples'), ' ', Timer()]
        pbar = ProgressBar(widgets = widgets, max_value = UnknownLength)
        pbar.start()

    # timer
    setup_duration = time.time() - setup_duration

    # samples counter
    samples_io_start = 0
    samples_io_end = 0
    inference_duration = time.time()
    
    # store results in df too
    resultdf = []
    with torch.no_grad():
        results = []
        for idx, (im, smiles, identifier, fname, fidx) in enumerate(infer_loader):
                
            # convert to half if requested
            if args_i.dtype == "fp16":
                im = im.half()
            elif args_i.dtype == "int8":
                im = im.to(torch.int8)

            # upload data
            im = im.to(device)
                
            # forward pass
            pred = model_fw(im)
            npred = np.squeeze(pred.cpu().numpy())

            # sample counter
            inc_buffer[0] += pred.shape[0]
            samples_io_end += pred.shape[0]

            # fill and append data frame
            results.append(pd.DataFrame({"score": npred, 
                                         "smiles": smiles, 
                                         "identifier": identifier, 
                                         "receptor": receptor_id,
                                         "filename": fname, 
                                         "fileindex": fidx}))

            # write file: we skip the idx = 0 one
            if ((idx + 1) % args_i.output_frequency == 0):
                tmpdf = pd.concat(results).sort_values(by=['score'], ascending=False).reset_index(drop=True)
                if args_i.write_intermediate_files:
                    tmpdf.to_csv(os.path.join(args_i.o, f"predictions_{samples_io_start}-{samples_io_end-1}_rank-{comm_rank}.csv"))
                samples_io_start = samples_io_end
                results = []
                resultdf.append(tmpdf)

            # update counter and report if requested
            if (idx % args_i.update_frequency == 0):
                if have_mpi:
                    samples_win.Lock(0, MPI.LOCK_SHARED)
                    samples_win.Flush(0)
                    samples_win.Accumulate(inc_buffer, 0, op=MPI.SUM)
                    samples_win.Unlock(0)
                else:
                    samples[0] += inc_buffer[0]
                inc_buffer[0] = 0

                if (comm_rank == 0):
                    if have_mpi:
                        # lock window and flush
                        samples_win.Lock(0, MPI.LOCK_EXCLUSIVE)
                        samples_win.Flush(0)
                    smp = np.asscalar(samples_buffer)
                    if have_mpi:
                        # unlock window
                        samples_win.Unlock(0)     
                        
                    # update pbar
                    pbar.update(smp)
        
        # final update
        if have_mpi:
            samples_win.Lock(0, MPI.LOCK_SHARED)
            samples_win.Flush(0)
            samples_win.Accumulate(inc_buffer, 0, op=MPI.SUM)
            samples_win.Unlock(0)
        else:
            samples[0] += inc_buffer[0]
        inc_buffer[0] = 0

    # timer
    inference_duration = time.time() - inference_duration

    # get perf metrics
    if have_mpi:
        arr = np.array(setup_duration, dtype = np.float32)
        setup_duration = np.asscalar(comm.allreduce(arr)) / float(comm_size)
        arr = np.array(inference_duration, dtype = np.float32)
        inference_duration = np.asscalar(comm.allreduce(arr)) / float(comm_size)
        if stage_handles:
            arr = np.array(stage_duration, dtype = np.float32)
            stage_duration = np.asscalar(comm.allreduce(arr)) / float(comm_size)
        
    # final concat
    if results:
        tmpdf = pd.concat(results).sort_values(by=['score'], ascending=False).reset_index(drop=True)
        if args_i.write_intermediate_files:
            tmpdf.to_csv(os.path.join(args_i.o, f"predictions_{samples_io_start}-{samples_io_end-1}_rank-{comm_rank}.csv"))
        samples_io_start = samples_io_end
        results = []
        resultdf.append(tmpdf)

    # final update
    if comm_rank == 0:
        pbar.update(np.asscalar(samples_buffer))

    # get samples
    samples_count = np.asscalar(samples_buffer)
    
    if comm_rank == 0:
        print(f"Setup time: {setup_duration}s")
        if stage_handles:
            print(f"From this, exposed staging time: {stage_duration}s")
        print(f"Processed {samples_count} samples in {inference_duration}s, throughput {samples_count/inference_duration} samples/s")

    # concat and rank
    resultdf = pd.concat(resultdf).sort_values(by=['score'], ascending=False).reset_index(drop=True)

    # take the top n
    if args_i.t > 0:
        projdf = resultdf.nlargest(args_i.t, columns=["score"], keep="all").copy()
    else:
        projdf = resultdf.copy()

    # gather pandas frames from all nodes:
    if have_mpi:
        allresults = comm.gather(projdf, 0)
        if comm_rank == 0:
            gresultdf = pd.concat(allresults)
    else:
        gresultdf = projdf

    if comm_rank == 0:
        if args_i.t > 0:
            # do one more filtering
            gresultdf = gresultdf.nlargest(args_i.t, columns=["score"], keep="all").reset_index(drop=True)

        # write output
        filename = os.path.join(args_i.o, f"top-{args_i.t}_ligand.csv")
        write_ligand_files(filename, gresultdf, file_per_ligand = False)

    # wait for node 0
    comm.barrier()
        
    # free stuff
    if have_mpi:
        samples_win.Free()


if __name__ == "__main__":

    # do this here
    torch.multiprocessing.set_start_method('forkserver')

    # parse args
    parser = ArgumentParser()
    parser.add_argument('-m', type=str, required=True, help='input directory for model')
    parser.add_argument('-i', type=str, required=True, help='input glob string for data')
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--stage_dir', type=str, default=None, help='Where to stage the files')
    parser.add_argument('--num_stage_workers', type=int, required=False, default=1)
    parser.add_argument('-trt', type=str, required=False, default=None)
    parser.add_argument('-dtype', type=str, choices=["fp32", "fp16", "int8"], required=False)
    parser.add_argument('-num_calibration_batches', type=int, default=1000, required=False)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--update_frequency', type=int, required=False, default=5)
    parser.add_argument('--output_frequency', type=int, required=False, default=20)
    parser.add_argument('--write_intermediate_files', type=bool, required=False, default=False)
    parser.add_argument('-t', type=int, required=False, default=-1)
    parser.add_argument('-d', type=int, required=False, default=0)
    parser.add_argument('-j', type=int, required=False, default=1)
    parser.add_argument('-b', type=int, required=False, default=64)
    pargs =  parser.parse_args()

    main(pargs)
