import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    """
    A dummy dataset class that returns a random sequence of tokens and its clone for ssl purpose.
    """
    def __init__(self, vocab_size, seq_len, datasize):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = datasize

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        # since we are using random ids we do not shift the target by one
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return ids, ids.clone()
