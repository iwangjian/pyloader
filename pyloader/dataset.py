# -*- coding: utf-8 -*-
from typing import Dict, List, TypeVar, Generic

T_co = TypeVar('T_co', covariant=True)
PAD_ = '<pad>'
PAD_IDX = 0


class Dataset(Generic[T_co]):
    """An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`batch_encode`
    """

    def batch_encode(self, batch_data: List[Dict]) -> Dict[str, List]:
        """
        Encode batch data from raw format to numeric format.
        Args: 
            batch_data (List[Dict]): A list of dicts
        Returns:
            A dict of lists
        """
        raise NotImplementedError


class TextDataset(Dataset):
    """
    Dataset wrapping for text.
    Args:
        vocab_file (str): vocab file path (default: `None`).
        max_seq_len (int, optional): maximum sequence length of the input text (default: `100`).
        max_vocab_size (int, optional): maximum size for loading the vocab words (default: `30000`).
        
    """
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
        """
        Encode batch data from raw format to numeric format.
        Args: 
            batch_data (List[Dict]): A list of dicts
        Returns:
            batch (Dict[str, List]]): A dict of lists
        """
        input_ids = []
        attention_mask = []
        labels = []
        for sample in batch_data:
            text = sample["input_text"]
            label = sample["label"]
            if self.stoi:
                text_ids = [self.stoi.get(w, PAD_IDX) for w in text.split()]
            else:
                text_ids = [int(w) for w in text.split()]
            text_masks = [1] * len(text_ids)
            if len(text_ids) > self.max_seq_len:
                input_id = text_ids[:self.max_seq_len]
                input_mask = text_masks[:self.max_seq_len]
            else:
                input_id = text_ids + [PAD_IDX] * (self.max_seq_len-len(text_ids))
                input_mask = text_masks + [0] * (self.max_seq_len-len(text_ids))
            input_ids.append(input_id)
            attention_mask.append(input_mask)
            labels.append(label)
        batch = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "label": labels
        }
        return batch
