# -*- coding: utf-8 -*-
from pyloader import DataReader
from pyloader import Dataset
from pyloader import DataCollator
from pyloader import DataLoader
import json
import numpy as np
import tensorflow as tf

# Step 1: Inherit the class `DataReader` and implement the method `read_file()`
class JsonLineDataReader(DataReader):
    """Data reader for reading files with the format of json line."""
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
        print("Reading data from [{}]".format(path))
        dict_samples = {}
        n_sample = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                json_sample = json.loads(line)
                n_sample += 1
                for k, v in json_sample.items():
                    if k == "post" or k == "response":
                        if not k in dict_samples:
                            dict_samples[k] = [v]
                        else:
                            dict_samples[k].append(v)
        print("Total data: {}".format(n_sample))
        return dict_samples, n_sample


# Step 2: Inherit the class `Dataset` and implement the method `batch_encode()`
class TextDataset(Dataset):
    """Dataset wrapping for text."""
    def __init__(self, vocab_file=None, max_seq_len=100, max_vocab_size=30000):
        self.max_seq_len = max_seq_len
        self.max_vocab_size = max_vocab_size
        self.stoi = {}
        self.itos = {}
        if vocab_file is not None:
            self.load_vocab(vocab_file)
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr):
                if idx < self.max_vocab_size:
                    w = line.strip().split('\t')[0]
                    self.stoi[w] = idx
                    self.itos[idx] = w
        assert len(self.stoi) == len(self.itos)
        assert len(self.stoi) <= self.max_vocab_size

    def batch_encode(self, batch_data):
        """Encode batch data from raw format to numeric format."""
        post_ids = []
        post_mask = []
        resp_ids = []
        resp_maks = []
        for k, data_list in batch_data.items():
            if k == "post" or k == "response":
                for text in data_list:
                    if self.stoi:
                        text_ids = [self.stoi.get(w, 0) for w in text.split()]
                    else:
                        text_ids = [int(w) for w in text.split()]
                    text_masks = [1] * len(text_ids)
                    if len(text_ids) > self.max_seq_len:
                        input_id = text_ids[:self.max_seq_len]
                        input_mask = text_masks[:self.max_seq_len]
                    else:
                        input_id = text_ids + [0] * (self.max_seq_len-len(text_ids))
                        input_mask = text_masks + [0] * (self.max_seq_len-len(text_ids))
                    if k == "post":
                        post_ids.append(input_id)
                        post_mask.append(input_mask)
                    else:
                        resp_ids.append(input_id)
                        resp_maks.append(input_mask)
        batch = {
            "post_ids": post_ids, 
            "post_mask": post_mask,
            "resp_ids": resp_ids,
            "resp_maks": resp_maks
        }
        return batch


# Step 3: Inherit the class `DataCollator` and implement the method `collate_batch()`
class TFDataCollator(DataCollator):
    """Simple Tensforflow-based data collator."""
    
    def collate_batch(self, features):
        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        if "label" in features and features["label"] is not None:
            first = features["label"][0]
            if type(first) is int:
                labels_npy = np.array([f for f in features["label"]], dtype=np.int)
            else:
                labels_npy = np.array([f for f in features["label"]], dtype=np.float)
            labels = tf.convert_to_tensor(labels_npy)
            batch = {"label": labels}
        else:
            batch = {}

        # Handling of all other possible attributes
        for k, v in features.items():
            if k not in ("label") and v is not None:
                batch_npy = np.array([f for f in features[k]], dtype=np.int)
                batch[k] = tf.convert_to_tensor(batch_npy)
        return batch


def main_test():
    data_dir = "data/chunk_files/"
    vocab_file = "data/vocab.txt"
    datareader = JsonLineDataReader(batch_size=16, shuffle=True)
    dataset = TextDataset(vocab_file, max_seq_len=60, max_vocab_size=30000)
    datacollator = TFDataCollator()

    # define a DataLoader
    train_loader = DataLoader(
        datareader, dataset, datacollator, data_dir,
        num_epoch=5, max_queue_size=10, shuffle=True
    )

    # iteratively load data for training
    with tf.device("GPU:0" if tf.test.is_gpu_available() else "CPU"):
        for batch_step, inputs in enumerate(train_loader):
            for k, v in inputs.items():
                # show batch data
                print("step: {}\t{}:\n{}".format(batch_step, k, inputs[k]))
            
                # define your code
                # ...


if __name__ == "__main__":
    main_test()
