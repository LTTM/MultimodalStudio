# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# This code is a modified version of the original code available at
#
#     https://github.com/autonomousvision/sdfstudio (commit 370902a10dbef08cb3fe4391bd3ed1e227b5c165)
#
# At the moment of this file creation, the original code is licensed under the Apache License, Version 2.0;
# You may obtain a copy of the Apache License, Version 2.0, at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova

"""
DataLoaders
"""
import concurrent.futures
import multiprocessing
from collections import defaultdict
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType

from cameras.pixel_samplers import PixelSampler
from data.datasets import BaseDataset


def loading_collate_fn(batch_list: List[Dict[str, TensorType]]) -> Dict[str, Dict[str, TensorType]]:
    """
    Collate function used to load and prepare a batch of images from the dataset.
    It stacks the frames and the camera indexes for each modality.

    Args:
        batch_list: List of frame data to collate. It is a list of dictionaries with length equal to len(dataset).
                    The dictionaries are of the form:
                        {
                            modality_name: {
                                'images': torch.Tensor of shape (H, W, C),
                                'index': torch.Tensor of shape (1,)
                            }
                        }
                    where C is the number of channels, H is the height and W is the width.
    Returns:
        staked_batch: A dictionary with the form:
                        {
                            modality_name: {
                                'images': torch.Tensor of shape (N, H, W, C),
                                'indexes': torch.Tensor of shape (N,)
                            }
                        }
                        where N is the number of images in the batch.
    """
    stacked_batch = {mod: defaultdict(list) for mod in batch_list[0].keys()}
    for elem in batch_list:
        for mod in elem.keys():
            if elem[mod]['images'] is not None:
                stacked_batch[mod]['indexes'].append(elem[mod]['index'])
                stacked_batch[mod]['images'].append(elem[mod]['images'])
    for mod in stacked_batch:
        stacked_batch[mod]['indexes'] = torch.stack(stacked_batch[mod]['indexes'])
        stacked_batch[mod]['images'] = torch.stack(stacked_batch[mod]['images'])
    return stacked_batch

def single_eval_collate_fn(batch_list: List[Dict[str, TensorType]]) -> Dict[str, Dict[str, TensorType]]:
    """
    Collate function used to load and prepare an evaluation batch from the dataset.
    The batch is composed by a single macro-frame, which includes all the modality frames of the same view.

    Args:
        batch_list: List of frame data to collate. It is a list of dictionaries with length equal to 1.
                    The dictionary is of the form:
                        {
                            modality_name: {
                                'images': torch.Tensor of shape (H, W, C),
                                'index': torch.Tensor of shape (1,)
                            }
                        }
                    where C is the number of channels, H is the height and W is the width.
    Returns:
        staked_batch: A dictionary with the form:
                        {
                            modality_name: {
                                'images': torch.Tensor of shape (1, H, W, C),
                                'indexes': torch.Tensor of shape (1,)
                            }
                        }
    """
    stacked_batch = {mod: defaultdict(list) for mod in batch_list[0].keys()}
    elem = batch_list[0]
    for mod in elem.keys():
        if elem[mod]['images'] is not None:
            stacked_batch[mod]['indexes'] = elem[mod]['index']
            stacked_batch[mod]['images'] = elem[mod]['images'].unsqueeze(0)
        else:
            stacked_batch[mod]['indexes'] = elem[mod]['index']
            stacked_batch[mod]['images'] = None
    return stacked_batch

class CacheDataloader(DataLoader):
    """Dataloader that implements GPU caching of frames contained in the dataset.
    The iterator returns the coordinates and the pixel values of the sampled pixels.

    Args:
        dataset: Dataset to sample from.
        pixel_sampler: The pixel sampler to use for sampling pixels from the frames.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        pixel_sampler: PixelSampler,
        collate_fn = loading_collate_fn,
        **kwargs,
    ):
        self.dataset = dataset
        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.pixel_sampler = pixel_sampler
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get("num_workers", 0)

        self.first_time = True

        print(f"Caching {len(self.dataset)} views.")
        if len(self.dataset) > 500:
            print("Warning: If you run out of memory, try reducing the number of images to sample from.")
        self.cached_collated_batch = self._get_collated_batch()

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        indices = list(range(len(self.dataset)))
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in results:
                batch_list.append(res.result())

        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        return collated_batch

    def __iter__(self):
        while True:
            coords, pixels = self.pixel_sampler.sample(self.cached_collated_batch)
            yield coords, pixels

class SingleViewDataloader(DataLoader):
    """Dataloader that loads a single frame of the dataset.
    The iterator returns the coordinates and the pixel values of the sampled pixels.

    Args:
        dataset: Dataset to sample from.
        pixel_sampler: The pixel sampler to use for sampling pixels from the frames.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: BaseDataset,
        pixel_sampler: PixelSampler,
        collate_fn = single_eval_collate_fn,
        view_list: List[int] = None,
        **kwargs,
    ):
        super().__init__(dataset=dataset, **kwargs)
        self.pixel_sampler = pixel_sampler
        self.collate_fn = collate_fn
        self.selected_views = [
            self.dataset.get_unique_views().index(i)
            for i in view_list
            if i in self.dataset.get_unique_views()
        ] if view_list is not None else list(range(len(self.dataset.get_unique_views())))
        self.next_iterator_index = None

    def __iter__(self):
        self.next_iterator_index = 0
        return self

    def __next__(self):
        if self.next_iterator_index == len(self.selected_views):
            self.next_iterator_index = 0
        view_id = self.selected_views[self.next_iterator_index]
        batch = [self.dataset.dynamic_get(view_id)]
        collated_batch = self.collate_fn(batch)
        coords, pixels = self.pixel_sampler.sample(collated_batch)
        self.next_iterator_index += 1
        return coords, pixels
