# -*- coding: utf-8 -*-
import sys
import random
import logging
import json

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)]
)

class DataReader(object):
    """
    Base data reader
    Args:
        batch_size (int, optional): how many samples per batch to load (default: `16`).
        shuffle (bool, optional): set to `True` to have the data shuffled at every file chunk (default: `False`).
        random_seed (int, optional): random seed (default: `42`).
    """
    def __init__(self, batch_size=16, shuffle=False, random_seed=42):
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
    
    def _check_data(self, dict_samples, n_sample):
        for key_name, data_list in dict_samples.items():
            if len(data_list) != n_sample:
                raise ValueError("Key: {}, length = {}, mismatch with n_sample = {}!".format(
                    key_name, len(data_list), n_sample))
    
    def _prepare_data(self, dict_samples, n_sample):
        """
        Load data into `self.batch_data_list` and build `self.batch_size_list`.
        Args:
            dict_samples (Dict[str, List]): input samples, type: a dict of lists.
            n_sample (int): the number of samples.
        """
        if self.shuffle:
            all_samples = [{} for _ in range(n_sample)]
            for key, data_list in dict_samples.items():
                for i in range(n_sample):
                    all_samples[i][key] = data_list[i]
            random.shuffle(all_samples)
            prep_samples = {}
            for sample in all_samples:
                for k, v in sample.items():
                    if not k in prep_samples:
                        prep_samples[k] = [v]
                    else:
                        prep_samples[k].append(v)
        else:
            prep_samples = dict_samples

        self.batch_data_list = []
        self.batch_size_list = []
        self.n_sample = n_sample
        
        remain_sample = n_sample
        while remain_sample > 0:
            self.batch_data_list.append({})
            active_size = min(remain_sample, self.batch_size)
            self.batch_size_list.append(active_size)
            remain_sample -= active_size
        self.n_batch = len(self.batch_size_list)

        for key, data_list in prep_samples.items():
            for batch_idx in range(self.n_batch):
                st_idx = batch_idx * self.batch_size
                ed_idx = st_idx + self.batch_size
                data_batch = list(data_list[st_idx: ed_idx])
                self.batch_data_list[batch_idx][key] = data_batch
    
    def read_file(self, path):
        """
        Read files from disk. This function needs to be implemented.
        Args:
            path (string): the file path.
        Returns:
            dict_samples (Dict[str, List]): input samples, type: a dict of lists.
            n_sample (int): the number of samples.
        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError

    def load(self, path, read_fn=None):
        if read_fn is None:
            read_fn = self.read_file
        
        dict_samples, n_sample = read_fn(path)
        self._check_data(dict_samples, n_sample)
        self._prepare_data(dict_samples, n_sample)
    
    def get_batch(self, batch_idx):
        local_data = self.batch_data_list[batch_idx]
        local_size = self.batch_size_list[batch_idx]
        return local_data, local_size


class JsonLineDataReader(DataReader):
    """
    Data reader for reading files with the format of json line.
    Args:
        batch_size (int, optional): how many samples per batch to load (default: `4`).
        shuffle (bool, optional): set to `True` to have the data shuffled at every file chunk (default: `False`).
        random_seed (int, optional): random seed (default: `42`).
    """
    def __init__(self, batch_size=4, shuffle=False, random_seed=42):
        super(JsonLineDataReader, self).__init__(
            batch_size,
            shuffle=shuffle,
            random_seed=random_seed
        )
    
    def read_file(self, path):
        """
        Read files from disk.
        Args:
            path (string): the file path.
        Returns:
            dict_samples (Dict[str, List]): input samples, type: a dict of lists.
            n_sample (int): the number of samples.
        """
        logging.info("Reading data from [{}]".format(path))
        dict_samples = {}
        n_sample = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                json_sample = json.loads(line)
                n_sample += 1
                for k, v in json_sample.items():
                    if not k in dict_samples:
                        dict_samples[k] = [v]
                    else:
                        dict_samples[k].append(v)
        logging.info("Total data: {}".format(n_sample))
        
        return dict_samples, n_sample
