import torch.nn as nn
from torch.nn import Sequential, Flatten

from src.base_model import BaseModel
import torch
import random


class MLP(BaseModel):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10, fixed_initialization=False, seed=42, init_scheme=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.lambda_1 = 1
        if fixed_initialization:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            self._initialize_weights(scheme=init_scheme)
            seed = random.randint(1, 100)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)


    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.lambda_1 * x
        x = self.fc2(x)
        return x

    def get_sequential(self):
        return Sequential(Flatten(), self.fc1, self.relu1, self.fc2)

    def _initialize_weights(self, scheme=None):
        if scheme is None or scheme == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif scheme == 'normal':
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc2.weight)