import torch

from machine_learning.Collate import ProteinCollate


def get_padded_data_loader(dataset, batch_size):
    collator = ProteinCollate()
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def get_dataloader(dataset, batch_size=256):
    return get_padded_data_loader(dataset, batch_size)

