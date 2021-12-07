from metrics.metric import Metric
from typing import Dict, Union
import torch


class GoalPredictionNLL(Metric):
    """
    Negative log likelihood loss for ground truth goal nodes under predicted goal log-probabilities.
    """
    def __init__(self, args: Dict):
        self.name = 'goal_pred_nll'

    def compute(self, predictions: Dict, ground_truth: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Compute goal prediction NLL loss.

        :param predictions: Dictionary with 'goal_log_probs': log probabilities over nodes for goal prediction
        :param ground_truth: Dictionary with 'evf_gt': Look up table with visited edges. Only the goal transition edges
        will be used by the loss.
        """
        # Unpack arguments
        goal_log_probs = predictions['goal_log_probs']
        gt_goals = ground_truth['evf_gt'][:, :, -1].bool()

        loss = -torch.sum(goal_log_probs[gt_goals]) / goal_log_probs.shape[0]

        return loss
