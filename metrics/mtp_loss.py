from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import min_ade, traj_nll


class MTPLoss(Metric):
    """
    MTP loss modified to include variances. Uses MSE for mode selection. Can also be used with
    Multipath outputs, with residuals added to anchors.
    """

    def __init__(self, args: Dict = None):
        """
        Initialize MTP loss
        :param args: Dictionary with the following (optional) keys
            use_variance: bool, whether or not to use variances for computing regression component of loss,
                default: False
            alpha: float, relative weight assigned to classification component, compared to regression component
                of loss, default: 1
        """
        self.use_variance = args['use_variance'] if args is not None and 'use_variance' in args.keys() else False
        self.alpha = args['alpha'] if args is not None and 'alpha' in args.keys() else 1
        self.beta = args['beta'] if args is not None and 'beta' in args.keys() else 1
        self.name = 'mtp_loss'

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute MTP loss
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode (log) probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """

        # Unpack arguments
        traj = predictions['traj']
        log_probs = predictions['probs']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth

        # Useful variables
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]
        pred_params = 5 if self.use_variance else 2

        # Masks for variable length ground truth trajectories
        masks = ground_truth['masks'] if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj.device)

        # Obtain mode with minimum ADE with respect to ground truth:
        errs, inds = min_ade(traj, traj_gt, masks)
        inds_rep = inds.repeat(sequence_length, pred_params, 1, 1).permute(3, 2, 0, 1)

        # Calculate MSE or NLL loss for trajectories corresponding to selected outputs:
        traj_best = traj.gather(1, inds_rep).squeeze(dim=1)

        if self.use_variance:
            l_reg = traj_nll(traj_best, traj_gt, masks)
        else:
            l_reg = errs

        # Compute classification loss
        l_class = - torch.squeeze(log_probs.gather(1, inds.unsqueeze(1)))

        loss = self.beta * l_reg + self.alpha * l_class
        loss = torch.mean(loss)

        return loss
