import torch.nn as nn
from src.base_model import BaseModel
import torch
import random
from torch.nn import Sequential, Flatten


class Deep_MLP(BaseModel):
    def __init__(self, input_size=28 * 28, hidden_size_1=256, hidden_size_2=128, hidden_size_3=64, output_size=10,
                 fixed_initialization=False, seed=42, init_scheme=None, assign_w=False, w_mats=(), which_act=0):
        super(Deep_MLP, self).__init__()
        if which_act == 0:
            act_func1 = nn.ReLU()
            act_func2 = nn.ReLU()
            act_func3 = nn.ReLU()
        elif which_act == 1:
            act_func1 = nn.LeakyReLU()
            act_func2 = nn.LeakyReLU()
            act_func3 = nn.LeakyReLU()
        elif which_act == 2:
            act_func1 = nn.GELU()
            act_func2 = nn.GELU()
            act_func3 = nn.GELU()
        else:
            raise NotImplementedError
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size_1, bias=False)
        self.act1 = act_func1
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        self.act2 = act_func2
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3, bias=False)
        self.act3 = act_func3
        self.fc4 = nn.Linear(hidden_size_3, output_size, bias=False)
        self.lambda_1 = 1
        self.lambda_2 = 1
        self.lambda_3 = 1
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
        if assign_w:
            with torch.no_grad():
                self.fc1.weight.copy_(torch.tensor(w_mats[0]))
                self.fc2.weight.copy_(torch.tensor(w_mats[1]))
                self.fc3.weight.copy_(torch.tensor(w_mats[2]))
                self.fc4.weight.copy_(torch.tensor(w_mats[3]))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x

    def get_sequential(self):
        return Sequential(Flatten(), self.fc1, self.act1, self.fc2, self.act2, self.fc3, self.act3, self.fc4)

    def _initialize_weights(self, scheme=None):
        if scheme is None or scheme == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.xavier_uniform_(self.fc4.weight)
        elif scheme == 'normal':
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc2.weight)
            nn.init.normal_(self.fc3.weight)
            nn.init.normal_(self.fc4.weight)