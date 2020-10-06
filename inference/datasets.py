import numpy as np
import pickle
import gzip as zip
import pickle
import sys
import os
from shutil import copyfile

#threading and queue
import threading, queue

# torch stuff
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
from PIL import Image

# feature gen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def shard_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    if (overall_end < 0):
        overall_end = len(dataset.filelist)
    # configure the dataset to only process the split workload
    per_worker = int(np.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


class CompressedMoleculesDataset(IterableDataset):
    
    def __init__(self, filelist, start = 0, end = -1,
                 max_retry_count = 5,
                 dtype = "fp32",
                 encoding = "images",
                 max_prefetch_count = 5):
        # store arguments
        self.filelist = filelist
        self.encoding = encoding
        self.start = start
        self.end = end
        self.max_retry_count = max_retry_count
        self.max_prefetch_count = max_prefetch_count

        # datatype
        self.dtype = torch.float32
        if dtype == "fp16":
            self.dtype = torch.float16
        elif dtype == "int8":
            self.dtype = torch.int8
                
        # set image transform
        self.transform = transforms.ToTensor()

        if not self.encoding == "images":
            from features.generateFeatures import smiles_to_image

        # set queue to not initialized
        self.initialized = False

    def update_filelist(self, filelist, start = 0, end = -1):
        if not self.initialized:
            self.start = start
            self.end = end
            self.filelist = filelist
        else:
            raise RuntimeError("Error, you cannot change the file list while the prefetcher is running")

    def __len__(self):
        return len(self.filelist)

    def _init_queue(self):
        # truncate file list with start and end
        self.filelist = self.filelist[self.start:self.end]
        self.file_queue = queue.Queue()
        for fname in self.filelist:
            self.file_queue.put((fname, 0))

        # set up queue
        self.prefetch_queue = queue.Queue(maxsize = self.max_prefetch_count)

        # lock logic
        self.prefetch_lock = threading.Lock()
        self.prefetch_stop = False
        
        # start prefetch
        self.prefetch_thread = threading.Thread(target = self._prefetch)
        self.prefetch_thread.start()


    def __del__(self):
        if self.initialized:
            with self.prefetch_lock:
                self.prefetch_stop = True
            self.prefetch_thread.join()


    def _get_file(self, fname):
        try:
            with zip.open(fname, mode='rb') as z:
                data = pickle.loads(z.read())
        except:
            data = None
        return (fname, data)

    def _prefetch(self): 
        while not self.file_queue.empty():
            fname, retry_count = self.file_queue.get()
            token = self._get_file(fname)
            if token[1] is not None:
                self.prefetch_queue.put(token)
            else:
                retry_count += 1
                if retry_count < self.max_retry_count:
                    # requeue
                    self.file_queue.put((fname, retry_count))
                else:
                    # error here
                    print(f"Error, unable to open {fname} after {retry_count} retries.")
                    # emplace nevertheless, check later
                    self.prefetch_queue.put(token)
            self.file_queue.task_done()
            with self.prefetch_lock:
                if self.prefetch_stop:
                    return
    
    def __iter__(self):
        if not self.initialized:
            self._init_queue()
        
        for _ in range(len(self.filelist)):
            fname, data = self.prefetch_queue.get()
            self.prefetch_queue.task_done()
            if data is not None:
                for fidx, item in enumerate(data):
                    folder = item[0]
                    identifier = item[1]
                    smiles = item[2]
                    if self.encoding == "images":
                        image = self.transform(item[3]).to(self.dtype)
                    else:
                        image = smiles_to_image(smiles, mol_computed = False)
                        image = self.transform(image).to(self.dtype)
                    yield image, smiles, identifier, fname, fidx
