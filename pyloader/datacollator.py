# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch


class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    def collate_batch(self) -> Dict[str, Any]:
        """
        Take a dict of samples from a Dataset and collate them into a batch.
        Returns:
            A dict of tensors
        """
        pass


class PyDataCollator(DataCollator):
    """
    Simple PyTorch-based data collator that:
    - simply collates batches of dict-like objects
    - performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
    - does not do any additional preprocessing
    i.e., Property names of the input object will be used as corresponding inputs to the model.
    """

    def collate_batch(self, features: Dict[str, List]) -> Dict[str, torch.Tensor]:
        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in features and features["label"] is not None:
            first = features["label"][0]
            if type(first) is int:
                labels = torch.tensor([f for f in features["label"]], dtype=torch.long)
            else:
                labels = torch.tensor([f for f in features["label"]], dtype=torch.float)
            batch = {"label": labels}
        else:
            batch = {}

        # Handling of all other possible attributes.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in features.items():
            if k not in ("label") and v is not None:
                batch[k] = torch.tensor([f for f in features[k]], dtype=torch.long)
        return batch
        