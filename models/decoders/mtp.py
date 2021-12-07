from models.decoders.decoder import PredictionDecoder
from models.decoders.utils import bivariate_gaussian_activation
import torch
import torch.nn as nn
from typing import Dict


class MTP(PredictionDecoder):

    def __init__(self, args):
        """
        Prediction decoder for MTP

        args to include:
        num_modes: int number of modes K
        op_len: int prediction horizon
        hidden_size: int hidden layer size
        encoding_size: int size of context encoding
        use_variance: Whether to output variance params along with mean predicted locations
        """

        super().__init__()

        self.agg_type = args['agg_type']
        self.num_modes = args['num_modes']
        self.op_len = args['op_len']
        self.use_variance = args['use_variance']
        self.op_dim = 5 if self.use_variance else 2

        self.hidden = nn.Linear(args['encoding_size'], args['hidden_size'])
        self.traj_op = nn.Linear(args['hidden_size'], args['op_len'] * self.op_dim * self.num_modes)
        self.prob_op = nn.Linear(args['hidden_size'], self.num_modes)

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, agg_encoding: torch.Tensor) -> Dict:
        """
        Forward pass for MTP
        :param agg_encoding: aggregated context encoding
        :return predictions: dictionary with 'traj': K predicted trajectories and
            'probs': K corresponding probabilities
        """
        h = self.leaky_relu(self.hidden(agg_encoding))
        batch_size = h.shape[0]
        traj = self.traj_op(h)
        probs = self.log_softmax(self.prob_op(h))
        traj = traj.reshape(batch_size, self.num_modes, self.op_len, self.op_dim)
        probs = probs.squeeze(dim=-1)
        traj = bivariate_gaussian_activation(traj) if self.use_variance else traj

        predictions = {'traj': traj, 'probs': probs}

        return predictions
