import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from functools import partial
class Template(nn.Module):
    def __init__(self):
        super(Template, self).__init__()
        pass

    def forward(self, text, class_center, class_std):
        pass
    # copy the parameters of the model
    def copy_parameters(self, model):
        for p, q in zip(self.parameters(), model.parameters()):
            p.data = q.data.clone()
class Baseline(Template):
    """
    Network to predict class mean and std from text description 
    """
    def __init__(self, image_size=512, text_input_size=512, embed_dim=64, device='cuda:0'):
        super(Baseline, self).__init__()
        self.mean = None 
        self.std = None 
        self.counter = 0
        self.previous_batch_size = 0
        self.mean = torch.zeros(image_size).to(device)
        self.std = torch.zeros(image_size).to(device)
    def update(self, mean, std):
        self.mean = mean.to(self.mean.device)
        self.std = std.to(self.mean.device)
    def forward(self, text, class_center, class_std):
        batch_size = text.shape[0]
        return self.mean.repeat(batch_size,1), self.std.repeat(batch_size,1)
class RandomBaseline():
    """
    Network to predict class mean and std from text description 
    """
    def __init__(self):
        pass
    def fit(self, X,Y):
        self.training_data = Y 
    def predict(self, X):
        perm = torch.tensor(np.random.choice(list(range(self.training_data.shape[0])), X.shape[0]))
        return self.training_data[perm]

class Block(nn.Module):
    """
    Network to predict class mean and std from text description 
    """
    def __init__(self, input_dim=512, output_dim=512,  activation=nn.GELU, layer_norm=nn.Identity, dropout=0.1, batch_norm=False):
        super(Block, self).__init__()
        self.activation = activation()
        self.norm = layer_norm(input_dim)

        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim) if batch_norm else nn.Identity(),
            activation(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = self.norm(x) 
        x = self.network(x)+x
        return x

class Linear(Template):
    """
    Network to predict class mean and std from text description 
    """
    def __init__(self, image_size=512, text_input_size=512, embed_dim=0, expansion_ratio=4, activation=nn.GELU, layer_norm=lambda x:x, dropout=0.1, batch_norm=False, imconditionning=True, target_name='mean'):
        super(Linear, self).__init__()
        self.activation = activation()
        self.embed_dim = embed_dim if embed_dim > 0 else image_size
        self.norm = layer_norm(text_input_size)
        self.imconditionning = imconditionning
        if self.imconditionning:
            self.image_net = nn.Sequential(
                nn.Linear(image_size, max(1, int(self.embed_dim*expansion_ratio))), 
                nn.BatchNorm1d(max(1, int(self.embed_dim*expansion_ratio))) if batch_norm else nn.Identity(),
                activation(),
                nn.Dropout(dropout),
            )
        self.text_net = nn.Sequential(
            nn.Linear(text_input_size, max(1, int(self.embed_dim*expansion_ratio))),
            nn.BatchNorm1d(max(1, int(self.embed_dim*expansion_ratio))) if batch_norm else nn.Identity(),
            activation(),
            nn.Dropout(dropout),
        )
        if target_name in ['mean', 'std']:
            self.to_out = nn.Linear(max(1, int(self.embed_dim*expansion_ratio)), image_size)
        elif target_name == 'covariance':
            self.to_out = nn.Linear(max(1, int(self.embed_dim*expansion_ratio)), image_size*image_size)
    def forward(self, text, image):
        text = self.text_net(self.norm(text))
        if self.imconditionning:
            image = self.image_net(self.norm(image))
            x = text+image
        else:
            x = text
        return self.to_out(x)
        

def get_net(net_name):
    if net_name == 'baseline':
        return Baseline
    elif 'linear' in net_name:
        expansion_ratio = int(net_name.split('_')[1]) if '_' in net_name else 4
        return partial(Linear, expansion_ratio=expansion_ratio)
    elif net_name == 'random_baseline':
        return RandomBaseline
    else:
        raise NotImplementedError