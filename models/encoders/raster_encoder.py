from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from positional_encodings import PositionalEncodingPermute2D
from typing import Dict


class RasterEncoder(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        CNN encoder for raster representation of HD maps and surrounding agent trajectories.

        args to include
            'backbone': str CNN backbone to use (resnet18, resnet34 or resnet50)
            'input_channels': int Size of scene features at each grid cell
            'use_positional_encoding: bool Whether or not to add positional encodings to final set of features
            'target_agent_feat_size': int Size of target agent state
        """

        super().__init__()

        # Anything more seems like overkill
        resnet_backbones = {'resnet18': resnet18,
                            'resnet34': resnet34,
                            'resnet50': resnet50}

        # Initialize backbone:
        resnet_model = resnet_backbones[args['backbone']](pretrained=False)
        conv1_new = nn.Conv2d(args['input_channels'], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modules = list(resnet_model.children())[:-2]
        modules[0] = conv1_new
        self.backbone = nn.Sequential(*modules)

        # Positional encodings:
        num_channels = 2048 if self.backbone == 'resnet50' else 512
        self.use_pos_enc = args['use_positional_encoding']
        if self.use_pos_enc:
            self.pos_enc = PositionalEncodingPermute2D(num_channels)

        # Linear layer to embed target agent representation.
        self.target_agent_encoder = nn.Linear(args['target_agent_feat_size'], args['target_agent_enc_size'])
        self.relu = nn.ReLU()

    def forward(self, inputs: Dict) -> Dict:

        """
        Forward pass for raster encoder
        :param inputs: Dictionary with
            target_agent_representation: torch.Tensor with target agent state, shape[batch_size, target_agent_feat_size]
            surrounding_agent_representation: Rasterized BEV representation, shape [batch_size, 3, H, W]
            map_representation: Rasterized BEV representation, shape [batch_size, 3, H, W]
        :return encodings: Dictionary with
            'target_agent_encoding': torch.Tensor of shape [batch_size, 3],
            'context_encoding': torch.Tensor of shape [batch_size, N, backbone_feat_dim]
        """

        # Unpack inputs:
        target_agent_representation = inputs['target_agent_representation']
        surrounding_agent_representation = inputs['surrounding_agent_representation']
        map_representation = inputs['map_representation']

        # Apply Conv layers
        rasterized_input = torch.cat((map_representation, surrounding_agent_representation), dim=1)
        context_encoding = self.backbone(rasterized_input)

        # Add positional encoding
        if self.use_pos_enc:
            context_encoding = context_encoding + self.pos_enc(context_encoding)

        # Reshape to form a set of features
        context_encoding = context_encoding.view(context_encoding.shape[0], context_encoding.shape[1], -1)
        context_encoding = context_encoding.permute(0, 2, 1)

        # Target agent encoding
        target_agent_enc = self.relu(self.target_agent_encoder(target_agent_representation))

        # Return encodings
        encodings = {'target_agent_encoding': target_agent_enc,
                     'context_encoding': {'combined': context_encoding,
                                          'combined_masks': torch.zeros_like(context_encoding[..., 0]),
                                          'map': None,
                                          'vehicles': None,
                                          'pedestrians': None,
                                          'map_masks': None,
                                          'vehicle_masks': None,
                                          'pedestrian_masks': None
                                          },
                     }
        return encodings
