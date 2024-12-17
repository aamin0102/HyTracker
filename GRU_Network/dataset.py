import os

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = os.listdir(data_dir)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_number = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_number)
        with open(file_path, "r") as file:
            lines = file.readlines()
        lines = [line.split() for line in lines]
        # Convert elements to integers
        data_int = [[int(num) for num in sublist] for sublist in lines]
        input_seq, output_seq = data_int[:6], data_int[6]
        return torch.tensor(input_seq, dtype=torch.float), torch.tensor(output_seq, dtype=torch.float)



