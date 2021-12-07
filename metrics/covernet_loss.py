from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import min_mse


class CoverNetLoss(Metric):
    """
    Purely computes the classification component of the MTP loss.
    """

    def __init__(self):
        """
        Initialize CoverNetLoss
        """
        self.name = 'mtp_loss'

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute MTP loss
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """

        # Unpack arguments
        traj = predictions['traj']
        probs = predictions['probs']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth

        # Useful variables
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]

        # Masks for variable length ground truth trajectories
        masks = ground_truth['masks'] if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj.device)

        # Obtain mode with minimum MSE with respect to ground truth:
        errs, inds = min_mse(traj, traj_gt, masks)

        # Calculate NLL loss for trajectories corresponding to selected outputs (assuming model uses log_softmax):
        loss = - torch.squeeze(probs.gather(1, inds.unsqueeze(1)))
        loss = torch.mean(loss)

        return loss
