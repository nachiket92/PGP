from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
import math
from typing import Dict, Tuple


# TODO (WiP): Test with different datasets, visualize results.
class PolylineSubgraphs(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        Polyline subgraph encoder from VectorNet (Gao et al., CVPR 2020).
        Has N encoder layers. Each layer encodes every feature in a polyline using an MLP with shared
        weights, followed by a permutation invariant aggregation operator (element-wise max used in the paper).
        Aggregated vector is concatenated with each independent feature encoding.
        Layer is repeated N times. Final encodings are passed through the permutation invariant
        aggregation operator to give polyline encodings.

        args to include
            'num_layers': int Number of repeated encoder layers
            'mlp_size':  int Width of MLP hidden layer
            'lane_feat_size': int Lane feature dimension
            'agent_feat_size': int Agent feature dimension

        """
        super().__init__()
        self.num_layers = args['num_layers']
        self.mlp_size = args['mlp_size']
        self.lane_feat_size = args['lane_feat_size']
        self.agent_feat_size = args['agent_feat_size']

        # Encoder layers

        """
        Note: I'm not completely sure if VectorNet uses different MLPs for agents, map polylines and map polygons.
        The paper doesn't seem to mention this clearly. However, agents and map polylines will typically have different 
        attribute features. At least the first linear layer has to be different. 
        Shouldn't affect the global attention aggregator. All final feats will have the same dimensions.
        """

        lane_encoders = [nn.Linear(self.lane_feat_size + 2, self.mlp_size)]
        for n in range(1, self.num_layers):
            lane_encoders.append(nn.Linear(self.mlp_size * 2, self.mlp_size))
        self.lane_encoders = nn.ModuleList(lane_encoders)

        target_agent_encoders = [nn.Linear(self.agent_feat_size + 2, self.mlp_size)]
        for n in range(1, self.num_layers):
            target_agent_encoders.append(nn.Linear(self.mlp_size * 2, self.mlp_size))
        self.target_agent_encoders = nn.ModuleList(target_agent_encoders)

        surrounding_vehicle_encoders = [nn.Linear(self.agent_feat_size + 2, self.mlp_size)]
        for n in range(1, self.num_layers):
            surrounding_vehicle_encoders.append(nn.Linear(self.mlp_size * 2, self.mlp_size))
        self.surrounding_vehicle_encoders = nn.ModuleList(surrounding_vehicle_encoders)

        surrounding_ped_encoders = [nn.Linear(self.agent_feat_size + 2, self.mlp_size)]
        for n in range(1, self.num_layers):
            surrounding_ped_encoders.append(nn.Linear(self.mlp_size * 2, self.mlp_size))
        self.surrounding_ped_encoders = nn.ModuleList(surrounding_ped_encoders)

        # Layer norm and relu
        self.layer_norm = nn.LayerNorm(self.mlp_size)
        self.relu = nn.ReLU()

    def forward(self, inputs: Dict) -> Dict:

        target_agent_feats = inputs['target_agent_representation']
        lane_feats = inputs['map_representation']['lane_node_feats']
        lane_masks = inputs['map_representation']['lane_node_masks']
        vehicle_feats = inputs['surrounding_agent_representation']['vehicles']
        vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']
        ped_feats = inputs['surrounding_agent_representation']['pedestrians']
        ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']

        # Encode target agent
        target_agent_feats = self.convert2vectornet_feat_format(target_agent_feats.unsqueeze(1))
        target_agent_enc, _ = self.encode(self.target_agent_encoders, target_agent_feats,
                                          torch.zeros_like(target_agent_feats))
        target_agent_enc = target_agent_enc.squeeze(1)

        # Encode lanes
        lane_feats = self.convert2vectornet_feat_format(lane_feats)
        lane_masks = lane_masks[:, :, :-1, :]
        lane_enc, lane_masks = self.encode(self.lane_encoders, lane_feats, lane_masks)

        # Encode surrounding agents
        vehicle_feats = self.convert2vectornet_feat_format(vehicle_feats)
        vehicle_masks = vehicle_masks[:, :, :-1, :]
        vehicle_enc, vehicle_masks = self.encode(self.surrounding_vehicle_encoders, vehicle_feats, vehicle_masks)
        ped_feats = self.convert2vectornet_feat_format(ped_feats)
        ped_masks = ped_masks[:, :, :-1, :]
        ped_enc, ped_masks = self.encode(self.surrounding_ped_encoders, ped_feats, ped_masks)

        # Return encodings
        encodings = {'target_agent_encoding': target_agent_enc,
                     'context_encoding': {'combined': None,
                                          'combined_masks': None,
                                          'map': lane_enc,
                                          'vehicles': vehicle_enc,
                                          'pedestrians': ped_enc,
                                          'map_masks': lane_masks,
                                          'vehicle_masks': vehicle_masks,
                                          'pedestrian_masks': ped_masks
                                          },
                     }

        return encodings

    def encode(self, encoder_layers: nn.ModuleList, input_feats: torch.Tensor,
               masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies encoding layers to a given set of input feats
        """
        masks = masks[..., 0]
        masks[masks == 1] = -math.inf

        encodings = input_feats
        for n in range(len(encoder_layers)):
            encodings = self.relu(self.layer_norm(encoder_layers[n](encodings)))
            encodings = encodings + masks.unsqueeze(-1)
            agg_enc, _ = torch.max(encodings, dim=2)
            encodings = torch.cat((encodings, agg_enc.unsqueeze(2).repeat(1, 1, encodings.shape[2], 1)), dim=3)
            encodings[encodings == -math.inf] = 0

        agg_encoding, _ = torch.max(encodings, dim=2)
        masks[masks == -math.inf] = 1

        return agg_encoding, masks[..., 0]

    @staticmethod
    def convert2vectornet_feat_format(feats: torch.Tensor) -> torch.Tensor:
        """
        Helper function to convert a tensor of node features to the vectornet format.
        By default the datasets return node features of the format [x, y, attribute feats...].
        Vectornet uses the following format [x, y, x_next, y_next, attribute_feats]
        :param feats: Tensor of feats, shape [batch_size, max_polylines, max_len, feat_dim]
        :return: Tensor of updated feats, shape [batch_size, max_polylines, max_len, feat_dim + 2]
        """
        xy = feats[:, :, :-1, :2]
        xy_next = feats[:, :, 1:, :2]
        attr = feats[:, :, :-1, 2:]
        feats = torch.cat((xy, xy_next, attr), dim=3)
        return feats
