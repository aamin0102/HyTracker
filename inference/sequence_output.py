import numpy as np
import torch
import torch.nn as nn

from GRU_Network.model import GRUModel


class SequenceModel:
    def __init__(self,input_size=4, output_size=4,  num_layers = 1, hidden_size = 64, num_item=6):
        self.num_item = num_item
        self.model = GRUModel(input_size, hidden_size, num_layers, output_size, num_gru=num_item)
        self._initialize()

    def _initialize(self):
        pretrained = torch.load('weights/gru.pth')
        self.model.load_state_dict(pretrained)
        self.model.eval()

    def get_output(self, bboxes):
        self.model.eval()
        bboxes = torch.tensor(bboxes, dtype=torch.float)
        bboxes = bboxes.unsqueeze(0)
        outputs = np.array(self.model.get_output(bboxes).detach().cpu())
        return outputs[0]

