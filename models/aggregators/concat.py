import torch
from models.aggregators.aggregator import PredictionAggregator
from typing import Dict


class Concat(PredictionAggregator):
    """
    Concatenates target agent encoding and all context encodings.
    Set of context encodings needs to be the same size, ideally with a well-defined order.
    """
    def __init__(self):
        super().__init__()

    def forward(self, encodings: Dict) -> torch.Tensor:
        """
        Forward pass for Concat aggregator
        """
        target_agent_enc = encodings['target_agent_encoding']
        context_enc = encodings['context_encoding']
        batch_size = target_agent_enc.shape[0]

        if context_enc['combined'] is not None:
            context_vec = context_enc['combined'].reshape(batch_size, -1)
        else:
            map_vec = context_enc['map'].reshape(batch_size, -1) if context_enc['map'] else torch.empty(batch_size, 0)
            vehicle_vec = context_enc['vehicles'].reshape(batch_size, -1) if context_enc['map'] \
                else torch.empty(batch_size, 0)
            ped_vec = context_enc['pedestrians'].reshape(batch_size, -1) if context_enc['pedestrians']\
                else torch.empty(batch_size, 0)
            context_vec = torch.cat((map_vec, vehicle_vec, ped_vec), dim=1)

        op = torch.cat((target_agent_enc, context_vec), dim=1)
        return op
