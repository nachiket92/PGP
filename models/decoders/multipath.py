from models.decoders.mtp import MTP
from models.decoders.utils import k_means_anchors
import torch
import torch.nn as nn
from datasets.interface import SingleAgentDataset
from typing import Dict


class Multipath(MTP):

    def __init__(self, args):
        """
        Prediction decoder for Multipath. Almost identical to MTP, but predicts residuals with respect to anchors,
        include the same arguments
        """

        super().__init__(args)
        self.anchors = nn.Parameter(torch.zeros(self.num_modes, self.op_len, 2), requires_grad=False)

    def generate_anchors(self, ds: SingleAgentDataset):
        """
        Function to initialize anchors
        :param ds: train dataset for single agent trajectory prediction
        """

        self.anchors = nn.Parameter(k_means_anchors(self.num_modes, ds))

    def forward(self, agg_encoding: torch.Tensor) -> Dict:
        """
        Forward pass for Multipath
        :param agg_encoding: aggregated context encoding
        :return predictions: dictionary with 'traj': K predicted trajectories and
            'probs': K corresponding probabilities
        """

        predictions = super().forward(agg_encoding)
        predictions['traj'][..., :2] += self.anchors.unsqueeze(0)

        return predictions
