# -*- coding: utf-8 -*-
import random
import logging
import time
import json


class DataReaderBase(object):
    """
    Base data reader
    Args:
        batch_size (int): Batch size.
        shuffle (bool): The flag denoting shuffle or not, defalut=``False``.
        random_seed (int): Random seed.
    """
    def __init__(self, batch_size, shuffle=False, random_seed=23):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.batch_data_list = []
        self.batch_size_list = []
        self.n_batch = 0    # number of batches
        self.n_sample = 0   # number of samples

    def __len__(self):
        return self.n_sample

    def get_batch(self, batch_idx):
        local_data = self.batch_data_list[batch_idx]
        local_size = self.batch_size_list[batch_idx]
        return local_data, local_size
    
    def check_data(self, dict_samples, n_sample):
        for key, data_list in dict_samples.items():
            if len(data_list) != n_sample:
                raise ValueError("Key: {}, length = {}, mismatch with n_sample = {}".format(
                    key, len(data_list), n_sample))
    
    def prepare_data(self, dict_samples, n_sample):
        """
        Load data into ``self.batch_data_list`` and build ``self.batch_size_list``.
        Args:
            dict_samples (Dict[List]): Input samples, type: a dict of lists.
            n_sample (int): The number of samples.
        """
        if self.shuffle:
            random.shuffle(dict_samples)

        self.n_sample = remain_sample = n_sample
        while remain_sample > 0:
            self.batch_data_list.append({})
            active_size = min(remain_sample, self.batch_size)
            self.batch_size_list.append(active_size)
            remain_sample -= active_size
        self.n_batch = len(self.batch_size_list)

        for key, data_list in dict_samples.items():
            for batch_idx in range(self.n_batch):
                st_idx = batch_idx * self.batch_size
                ed_idx = st_idx + self.batch_size
                data_batch = list(data_list[st_idx: ed_idx])
                self.batch_data_list[batch_idx][key] = data_batch
    
    def read_file(self, path):
        """
        Read a file from file system. This function needs to be implemented.
        Args:
            path (string): The file path.
        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError

    def load(self, path, read_fn=None):
        if read_fn is None:
            read_fn = self.read_file
        
        dict_samples, n_sample = read_fn(path)
        self.check_data(dict_samples, n_sample)
        self.prepare_data(dict_samples, n_sample)


class JsonLineDataReader(DataReaderBase):
    """
    Data reader for reading files with the format of json line.
    Args:
        batch_size (int): Batch size.
        shuffle (bool): The flag denoting shuffle or not, defalut=``False``.
        random_seed (int): Random seed.
    """
    def __init__(self, batch_size, shuffle=False, random_seed=23):
        super(JsonLineDataReader, self).__init__(
            batch_size,
            shuffle=shuffle,
            random_seed=random_seed
        )
    
    def read_file(self, path):
        """
        Read a file from file system.
        Args:
            path (string): The file path.
        Returns:
            dict_samples (Dict[List]): Input samples, type: a dict of lists.
            n_sample (int): The number of samples.
        """
        logging.info("Reading data from [{}]".format(path))
        dict_samples = {}
        n_sample = 0
        with open(path, 'r') as f:
            for line in f:
                json_sample = json.loads(line)
                n_sample += 1
                for k, v in json_sample:
                    if not k in dict_samples:
                        dict_samples[k] = [v]
                    else:
                        dict_samples[k].append(v)
        logging.info("Total data: {}".format(n_sample))
        
        return dict_samples, n_sample
