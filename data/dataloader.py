from .match_sampler import MatchingSampler
from .distributed import DistributedSampler
from .random import RandomSampler
import torch



def get_dataloader(dataset, batch_size, n_frames, n_shops, is_parallel):
    if is_parallel:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = RandomSampler(dataset)

    batch_sampler = MatchingSampler(dataset, sampler, batch_size, n_frames, n_shops, drop_last=True)

    data_loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_sampler=batch_sampler)
    # print("%d %d" % (rank, len(list(data_loader))))
    return data_loader
