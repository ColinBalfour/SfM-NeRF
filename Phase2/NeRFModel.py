import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, embed_direction_L=4, num_channels=256, pos_encoding=True):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        
        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L
        self.pos_encoding = pos_encoding
        
        pos_input = 3 * (2 * embed_pos_L + 1)
        dir_input = 3 * (2 * embed_direction_L + 1)
        
        if not pos_encoding:
            pos_input = 3
            dir_input = 3
        
        self.fc1 = nn.Sequential(
            nn.Linear(pos_input, 256),
            nn.ReLU(),
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(num_channels, num_channels), # 2
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 3
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 4
            nn.ReLU(),
        )
        
        self.skip_layer = nn.Sequential(
            nn.Linear(num_channels + pos_input, num_channels), # 5
            nn.ReLU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(num_channels, num_channels), # 6
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 7
            nn.ReLU(),
            nn.Linear(num_channels, num_channels + 1), # 8
            nn.ReLU(),
        )
        
        self.density_fc = nn.Linear(num_channels, 1)
        
        self.rgb_fc = nn.Sequential(
            nn.Linear(num_channels + dir_input, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        
        

    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################
        
        x_encoded = [x]
        for l_pos in range(L):
            x_encoded.append(torch.sin(2**l_pos * torch.pi * x))
            x_encoded.append(torch.cos(2**l_pos * torch.pi * x))

        return torch.cat(x_encoded, dim=-1)

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################
        
        if self.pos_encoding:
            pos = self.position_encoding(pos, self.embed_pos_L)
            direction = self.position_encoding(direction, self.embed_direction_L)
        
        out = self.fc1(pos)
        out = self.block1(out)
        out = self.skip_layer(torch.cat([out, pos], dim=-1))
        out = self.block2(out)
        
        density = out[:, :, 0].unsqueeze(-1)
        out = out[:, :, 1:]
        
        # density = self.density_fc(out)
        # density = F.softplus(self.density_fc(out)) - 1e-2
        
        dir_input = torch.cat((out, direction), dim=-1)
        rgb = self.rgb_fc(dir_input)

        return density, rgb
