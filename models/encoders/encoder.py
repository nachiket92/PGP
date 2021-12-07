import torch.nn as nn
import abc
from typing import Dict


class PredictionEncoder(nn.Module):
    """
    Base class for encoders for single agent prediction.
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, inputs: Dict) -> Dict:
        """
        Abstract method for forward pass. Returns dictionary of encodings. Should typically include
        1) target agent encoding, 2) context encoding: encodes map and surrounding agents.

        Context encodings will typically be a set of features (agents or parts of the map),
        with shape: [batch_size, set_size, feature_dim],
        sometimes along with masks for some set elements to account for varying set sizes

        :param inputs: Dictionary with
            'target_agent_representation': target agent history
            'surrounding_agent_representation': surrounding agent history
            'map_representation': HD map representation
        :return encodings: Dictionary with input encodings
        """
        raise NotImplementedError()
