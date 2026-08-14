import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PoetryDataset(Dataset):
    def __init__(self, sentences, char_to_idx):
        self.sentences = sentences
        self.char_to_idx = char_to_idx
        self.bos_id = char_to_idx["<BOS>"]
        self.eos_id = char_to_idx["<EOS>"]
        self.unk_id = char_to_idx.get("<UNK>")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        if self.unk_id is None:
            char_ids = [self.char_to_idx[char] for char in sentence]
        else:
            char_ids = [self.char_to_idx.get(char, self.unk_id) for char in sentence]
        input_ids = [self.bos_id] + char_ids
        target_ids = char_ids + [self.eos_id]
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        return input_tensor, target_tensor

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=0,
    )
    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=0,
    )
    return inputs, targets