# -*- coding: utf-8 -*-
import os
import sys
import random
import logging
import time
import copy
from queue import Queue
from threading import Thread
from typing import TypeVar, Generic
from .datareader import DataReader
from .dataset import Dataset
from .datacollator import DataCollator

MAX_QUEUE_SIZE = 10
T_co = TypeVar('T_co', covariant=True)

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)]
)

class DataLoader(Generic[T_co]):
    """
    Data loader. Provides an iterable over the given big dataset.
    
    Args:
        datareader (DataReader): datareader from which to load the data.
        dataset (Dataset): encode data to numeric format.
        datacollator (DataCollator): collate batch data into tensors.
        root_dir (str): root directory of the data files.
        num_epoch (int): number of training epochs (default: `1`)
        max_queue_size (int): maximum size of the queue for maintaining datareaders (default: `20`).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
        random_seed (int): random seed (default: `42`).
    """

    def __init__(self, 
                datareader: DataReader,
                dataset: Dataset,
                datacollator: DataCollator,
                root_dir: str, 
                num_epoch: int = 1, 
                max_queue_size: int = 10, 
                shuffle: bool = False, 
                random_seed: int = 42):

        if not isinstance(datareader, DataReader):
            raise TypeError("datareader should be a valid `pyloader.DataReader` type!")
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset should be a valid `pyloader.Dataset` type!")
        if not isinstance(datacollator, DataCollator):
            raise TypeError("datacollator should be a valid `pyloader.DataCollator` type!")
        
        self.file_chunk_list = []
        self._check_files(root_dir)

        self.datareader = datareader
        self.dataset = dataset
        self.datacollator = datacollator
        self.num_epoch = num_epoch
        self.current_epoch = 0

        self.max_queue_size = MAX_QUEUE_SIZE
        if max_queue_size > 0 and max_queue_size < 1000:
            self.max_queue_size = max_queue_size
        else:
            raise Warning("max_queue_size is suggested to be in the range (0, 1000). Set as default to {}.".format(self.max_queue_size))
        
        self.shuffle = shuffle
        self.random_seed = random_seed
        random.seed(self.random_seed)
        
        self.reader_queue = Queue(maxsize=self.max_queue_size)
        self.reader_queue_size = 0
        self.reader_iter = self._reader_generator()
        self.batch_iter = self._batch_generator()

        # Start a thread that monitors the reader queue
        self.reader_q_thread = Thread(target=self._push_reader_queue)
        self.reader_q_thread.setDaemon(True)
        self.reader_q_thread.start()

        # Start a thread that monitors the other threads and restarts them if they're dead
        self.watch_thread = Thread(target=self._monitor_threads)
        self.watch_thread.setDaemon(True)
        self.watch_thread.start()

        # Initialize reader queue to be full
        self._init_queue()
    
    def _check_files(self, root_dir):
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                fpath = os.path.join(root, f)
                if os.path.isfile(fpath):
                    self.file_chunk_list.append(fpath)
                else:
                    raise FileExistsError("File `{}` not exist!".format(fpath))
    
    def _init_queue(self):
        while True:
            if self.reader_queue_size < self.max_queue_size:
                time.sleep(2)
            else:
                break

    def _batch_generator(self):
        """Batch data iterator."""
        while self.reader_queue_size > 0:
            data_reader = self.reader_queue.get()
            assert isinstance(data_reader, DataReader)
            n_batch = data_reader.n_batch
            self.reader_queue_size -= 1
            for batch_idx in range(n_batch):
                batch_data, _ = data_reader.get_batch(batch_idx)
                batch_vars = self.dataset.batch_encode(batch_data)
                batch_tensors = self.datacollator.collate_batch(batch_vars)
                yield batch_tensors

    def _reader_generator(self):
        """Datareader iterator."""
        for epoch in range(1, self.num_epoch + 1):
            logging.info("Epoch: {}".format(epoch))
            self.current_epoch = epoch
            if self.shuffle:
                random.shuffle(self.file_chunk_list)
            for f_path in self.file_chunk_list:
                data_reader = copy.deepcopy(self.datareader)
                data_reader.load(path=f_path)
                yield data_reader

    def _push_reader_queue(self):
        """Push data reader into the reader queue when it's not full."""
        while True:
            if self.reader_queue_size <= self.max_queue_size:
                try:
                    data_reader = next(self.reader_iter)
                    self.reader_queue.put(data_reader)
                    self.reader_queue_size += 1
                except StopIteration:
                    break

    def _monitor_threads(self):
        """Watch loader queue thread and restart if dead."""
        while True:
            time.sleep(60)
            if not self.reader_q_thread.is_alive():  # if the thread is dead
                logging.info("Found loader queue thread dead. Restarting...")
                new_t = Thread(target=self.push_reader_queue)
                self.reader_q_thread = new_t
                new_t.daemon = True
                new_t.start()
                logging.info("Loader queue thread started.")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Batch data iterator.
        Returns:
            batch_data ([Dict[Tensor]]): A dictionary of tensors
        """
        try:
            batch_data = next(self.batch_iter)
            return batch_data
        except StopIteration:
            raise StopIteration
    
    def get_batch(self):
        """
        Batch data iterator.
        Returns:
            batch_data ([Dict[Tensor]]): A dictionary of tensors
        """
        try:
            batch_data = next(self.batch_iter)
            return batch_data
        except StopIteration:
            raise StopIteration

    def get_epoch(self):
        return self.current_epoch
