import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class NeRFmodel(nn.Module):
#     def __init__(self, embed_pos_L=10, embed_direction_L=4, num_channels=256, pos_encoding=True):
#         super(NeRFmodel, self).__init__()
#         #############################
#         # network initialization
#         #############################
        
#         self.embed_pos_L = embed_pos_L
#         self.embed_direction_L = embed_direction_L
#         self.pos_encoding = pos_encoding
        
#         pos_input = 3 * (2 * embed_pos_L + 1)
#         dir_input = 3 * (2 * embed_direction_L + 1)
        
#         if not pos_encoding:
#             pos_input = 3
#             dir_input = 3
        
#         self.fc1 = nn.Sequential(
#             nn.Linear(pos_input, 256),
#             nn.ReLU(),
#         )
        
#         self.block1 = nn.Sequential(
#             nn.Linear(num_channels, num_channels), # 2
#             nn.ReLU(),
#             nn.Linear(num_channels, num_channels), # 3
#             nn.ReLU(),
#             nn.Linear(num_channels, num_channels), # 4
#             nn.ReLU(),
#         )
        
#         self.skip_layer = nn.Sequential(
#             nn.Linear(num_channels + pos_input, num_channels), # 5
#             nn.ReLU(),
#         )
        
#         self.block2 = nn.Sequential(
#             nn.Linear(num_channels, num_channels), # 6
#             nn.ReLU(),
#             nn.Linear(num_channels, num_channels), # 7
#             nn.ReLU(),
#             nn.Linear(num_channels, num_channels + 1), # 8
#             nn.ReLU(),
#         )
        
#         self.density_fc = nn.Linear(num_channels, 1)
        
#         self.rgb_fc = nn.Sequential(
#             nn.Linear(num_channels + dir_input, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3),
#             nn.Sigmoid()
#         )
        
        

#     def position_encoding(self, x, L):
#         #############################
#         # Implement position encoding here
#         #############################
        
#         x_encoded = [x]
#         for l_pos in range(L):
#             x_encoded.append(torch.sin(2**l_pos * torch.pi * x))
#             x_encoded.append(torch.cos(2**l_pos * torch.pi * x))

#         return torch.cat(x_encoded, dim=-1)

#     def forward(self, pos, direction):
#         #############################
#         # network structure
#         #############################
        
#         if self.pos_encoding:
#             pos = self.position_encoding(pos, self.embed_pos_L)
#             direction = self.position_encoding(direction, self.embed_direction_L)
        
#         out = self.fc1(pos)
#         out = self.block1(out)
#         out = self.skip_layer(torch.cat([out, pos], dim=-1))
#         out = self.block2(out)
        
#         density = out[:, :, 0].unsqueeze(-1)
#         out = out[:, :, 1:]
        
#         # density = self.density_fc(out)
#         # density = F.softplus(self.density_fc(out)) - 1e-2
        
#         dir_input = torch.cat((out, direction), dim=-1)
#         rgb = self.rgb_fc(dir_input)

#         return density, rgb

class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, embed_direction_L=4, num_channels=256, pos_encoding=True):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################

        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L
        self.pos_encoding = pos_encoding

        # Calculate input dimensions based on positional encoding
        pos_input = 3 * (2 * embed_pos_L + 1) if pos_encoding else 3
        dir_input = 3 * (2 * embed_direction_L + 1) if pos_encoding else 3

        # First layer processes encoded position
        self.fc1 = nn.Linear(pos_input, num_channels)

        # Main MLP for processing position
        self.fc_layers = nn.ModuleList([
            nn.Linear(num_channels, num_channels) for _ in range(4)
        ])

        # Skip connection layer
        self.skip_layer = nn.Linear(num_channels + pos_input, num_channels)

        # Additional layers after skip connection
        self.fc_layers2 = nn.ModuleList([
            nn.Linear(num_channels, num_channels) for _ in range(3)
        ])

        # Density output
        self.density_layer = nn.Linear(num_channels, 1)

        # Feature vector output for color prediction
        self.feature_layer = nn.Linear(num_channels, num_channels)

        # Color prediction layers
        self.rgb_layer1 = nn.Linear(num_channels + dir_input, 128)
        self.rgb_layer2 = nn.Linear(128, 3)

    def position_encoding(self, x, L):
        """
        Apply positional encoding to input x
        Args:
            x: tensor of shape [..., 3]
            L: number of frequency bands
        Returns:
            encoded: tensor of shape [..., 3 * (2 * L + 1)]
        """
        encoded = [x]
        for i in range(L):
            freq = 2.0 ** i
            for func in [torch.sin, torch.cos]:
                encoded.append(func(freq * np.pi * x))
        return torch.cat(encoded, dim=-1)

    def forward(self, pos, direction):
        """
        Forward pass through the network
        Args:
            pos: tensor of shape [..., 3] - 3D positions
            direction: tensor of shape [..., 3] - viewing directions
        Returns:
            density: tensor of shape [..., 1] - density at each position
            rgb: tensor of shape [..., 3] - RGB color at each position
        """
        # Apply positional encoding if enabled
        if self.pos_encoding:
            pos_encoded = self.position_encoding(pos, self.embed_pos_L)
            dir_encoded = self.position_encoding(direction, self.embed_direction_L)
        else:
            pos_encoded = pos
            dir_encoded = direction

        # Initial layer
        h = F.relu(self.fc1(pos_encoded))

        # First block of fully-connected layers
        for i, layer in enumerate(self.fc_layers):
            h = F.relu(layer(h))
            # Skip connection at the 4th layer
            if i == 3:
                h = torch.cat([h, pos_encoded], dim=-1)
                h = F.relu(self.skip_layer(h))

        # Second block
        for layer in self.fc_layers2:
            h = F.relu(layer(h))

        # Density
        density = F.softplus(self.density_layer(h))

        # Feature vector for color prediction
        feature = F.relu(self.feature_layer(h))

        # Combine feature vector with viewing direction for color prediction
        rgb_input = torch.cat([feature, dir_encoded], dim=-1)
        rgb = F.relu(self.rgb_layer1(rgb_input))
        rgb = torch.sigmoid(self.rgb_layer2(rgb))  # Sigmoid to ensure [0,1] range

        return density, rgb
