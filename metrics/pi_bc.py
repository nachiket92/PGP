from metrics.metric import Metric
from typing import Dict, Union
import torch


class PiBehaviorCloning(Metric):
    """
    Behavior closing loss for training graph traversal policy.
    """
    def __init__(self, args: Dict):
        self.name = 'pi_bc'

    def compute(self, predictions: Dict, ground_truth: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Compute negative log likelihood of ground truth traversed edges under learned policy.

        :param predictions: Dictionary with 'pi': policy for lane graph traversal (log probabilities)
        :param ground_truth: Dictionary with 'evf_gt': Look up table with visited edges
        """
        # Unpack arguments
        pi = predictions['pi']
        evf_gt = ground_truth['evf_gt']

        loss = -torch.sum(pi[evf_gt.bool()]) / pi.shape[0]

        return loss
