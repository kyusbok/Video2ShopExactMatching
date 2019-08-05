import random
from torch.utils.data.sampler import BatchSampler
import torch
from torch._six import int_classes as _int_classes
from torch.utils.data import Sampler


class MatchingSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, dataset, sampler, batch_size, n_frames, n_shops, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.data = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.n_shops = n_shops
        self.drop_last = drop_last
        self.customer_inds = self.data.street_inds
        self.shop_inds = self.data.shop_inds
        self.customer_used = torch.zeros((len(self.customer_inds, )))
        self.shop_used = torch.zeros((len(self.shop_inds, )))
        self.match_map_shop = self.data.match_map_shop
        self.match_map_street = self.data.match_map_street
        self.pair_keys = [k for k in self.data.match_map_street.keys()]
        self.seed_dict = {}
        self.tmp_index = []

    def __iter__(self):
        batch = []
        # alternate = False
        for idx in self.sampler:
            batch.append((self.pair_keys[idx], "shop", None))
            tmp_video_samples = sorted([random.random() for x in range(self.n_frames)])

            for t in tmp_video_samples:
                batch.append((self.pair_keys[idx], "street", t))
            if self.batch_size == 1 or len(batch) == self.batch_size:
                yield batch
                batch = []
            # alternate = not alternate
        if not self.drop_last:
            yield batch
            batch = []

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.n_shops
        else:
            return len(self.sampler) // self.n_shops + 1

    def _getTypeInds(self, type_s):
        inds = []
        N = len(self.data)
        for i in range(1, N + 1):
            if self.data.coco.imgs[i]['source'] == type_s:
                inds.append(i)

        return inds

    def _getSamePairInShop(self, id):
        match_desc = self.data.coco.imgs[id]['match_desc']
        ids = []

        for x in match_desc:
            hashable_key = x + '_' + str(match_desc.get(x))
            matches = self.match_map_shop.get(hashable_key)
            if matches is not None:
                ids = ids + matches

        return ids

    def _getSamePairInStreet(self, id):
        match_desc = self.data.coco.imgs[id]['match_desc']
        ids = []

        for x in match_desc:
            hashable_key = x + '_' + str(match_desc.get(x))
            matches = self.match_map_street.get(hashable_key)
            if matches is not None:
                ids = ids + matches

        return ids
