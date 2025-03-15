import torch
import torch.nn as nn
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L, num_channels=256):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        
        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L
        
        self.fc1 = nn.Sequential(
            nn.Linear(embed_pos_L, 256),
            nn.ReLU(),
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(num_channels, num_channels), # 2
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 3
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 4
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 5
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 6
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 7
            nn.ReLU(),
            nn.Linear(num_channels, num_channels), # 8
            nn.ReLU()
        )
        
        self.density_fc = nn.Linear(num_channels, 1)
        
        self.features_fc = nn.Sequential(
            nn.Linear(num_channels + embed_direction_L, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        
        

    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################

        return 

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################
        
        out = self.fc1(pos)
        out = self.block1(out)
        density = self.density_fc(out)
        
        dir_input = torch.cat((out, direction), dim=-1)
        rgb = self.features_fc(dir_input)

        return density, rgb
