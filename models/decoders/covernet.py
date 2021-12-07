from models.decoders.decoder import PredictionDecoder
from models.decoders.utils import k_means_anchors
import torch
import torch.nn as nn
from datasets.interface import SingleAgentDataset
from typing import Dict


class CoverNet(PredictionDecoder):

    def __init__(self, args):
        """
        Prediction decoder for CoverNet

        args to include:
        num_modes: int number of modes K
        op_len: int prediction horizon
        hidden_size: int hidden layer size
        encoding_size: int size of context encoding
        """

        super().__init__()

        self.agg_type = args['agg_type']
        self.num_modes = args['num_modes']
        self.hidden = nn.Linear(args['encoding_size'], args['hidden_size'])
        self.op_len = args['op_len']
        self.prob_op = nn.Linear(args['hidden_size'], self.num_modes)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.anchors = nn.Parameter(torch.zeros(self.num_modes, self.op_len, 2), requires_grad=False)

    def generate_anchors(self, ds: SingleAgentDataset):
        """
        Function to initialize anchors. Extracts fixed trajectory set with k-means. Dynamic trajectory sets
        have not been implemented.
        :param ds: train dataset for single agent trajectory prediction
        """
        self.anchors = nn.Parameter(k_means_anchors(self.num_modes, ds))

    def forward(self, agg_encoding: torch.Tensor) -> Dict:
        """
        Forward pass for CoverNet
        :param agg_encoding: aggregated context encoding
        :return predictions: dictionary with 'traj': K predicted trajectories and
            'probs': K corresponding probabilities
        """
        h = self.leaky_relu(self.hidden(agg_encoding))
        batch_size = h.shape[0]
        probs = self.log_softmax(self.prob_op(h))
        probs = probs.squeeze(dim=-1)
        traj = self.anchors.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        predictions = {'traj': traj, 'probs': probs}

        return predictions
