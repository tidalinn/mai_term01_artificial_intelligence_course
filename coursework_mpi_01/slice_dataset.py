import torch
import torch.utils.data as data_utils

def slice_dataset(rank, ranks_total, dataset):
    slice_step = len(dataset) / ranks_total
    
    if rank == 0:
        indices = torch.arange(0, slice_step)
    else:
        indices = torch.arange(slice_step * rank, slice_step * rank + slice_step)
        
    return data_utils.Subset(dataset, indices)