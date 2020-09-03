import numpy as np
import pickle
import glob
import gzip as zip
import pickle

#threading and queue
import threading, queue

# torch stuff
import torch
from torch.utils.data import IterableDataset
from PIL import Image
#import torchvision

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


class CompressedImageDataset(IterableDataset):
    
    def __init__(self, globstring, start = 0, end = -1,
                 max_prefetch_count = 5):
        # store arguments
        self.globstring = globstring
        self.start = start
        self.end = end
        self.max_prefetch_count = max_prefetch_count
        
        # get filelist
        self.filelist = glob.glob(self.globstring)
        
        # set queue to not initialized
        self.initialized = False

    def _init_queue(self):
        # truncate file list with start and end
        self.filelist = self.filelist[self.start:self.end]

        # set up queue
        self.prefetch_queue = queue.Queue(maxsize = self.max_prefetch_count)

        # lock logic
        self.prefetch_lock = threading.Lock()
        self.prefetch_stop = False
        
        # start prefetch
        self.prefetch_thread = threading.Thread(target=self._prefetch)
        self.prefetch_thread.start()


    def __del__(self):
        if self.initialized:
            with self.prefetch_lock:
                self.prefetch_stop = True
            self.prefetch_thread.join()


    def _get_file(self, fname):
        with zip.open(fname, mode='rb') as z:
            data = pickle.loads(z.read())
        return data

    def _prefetch(self):
        for fname in self.filelist:
            data = self._get_file(fname)
            self.prefetch_queue.put(data)
            with self.prefetch_lock:
                if self.prefetch_stop:
                    return
            
    def __iter__(self):
        if not self.initialized:
            self._init_queue()

        data = self.prefetch_queue.get()
        for item in data:
            folder = item[0]
            identifier = item[1]
            image = np.asarray(item[3]).copy()
            yield image, identifier
