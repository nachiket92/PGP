import abc
import torch.utils.data as torch_data
import numpy as np
from typing import Union, Dict
import os


class TrajectoryDataset(torch_data.Dataset):
    """
    Base class for trajectory datasets.
    """

    def __init__(self, mode: str, data_dir: str):
        """
        Initialize trajectory dataset.
        :param mode: Mode of operation of dataset
        :param data_dir: Directory to store extracted pre-processed data
        """
        if mode not in ['compute_stats', 'extract_data', 'load_data']:
            raise Exception('Dataset mode needs to be one of {compute_stats, extract_data or load_data}')
        self.mode = mode
        self.data_dir = data_dir
        if mode != 'load_data' and not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns size of dataset
        """
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Union[Dict, int]:
        """
        Get data point, based on mode of operation of dataset.
        :param idx: data index
        """
        if self.mode == 'compute_stats':
            return self.compute_stats(idx)
        elif self.mode == 'extract_data':
            self.extract_data(idx)
            return 0
        else:
            return self.load_data(idx)

    @abc.abstractmethod
    def compute_stats(self, idx: int) -> Dict:
        """
        Function to compute dataset statistics like max surrounding agents, max nodes, max edges etc.
        :param idx: data index
        """
        raise NotImplementedError()

    def extract_data(self, idx: int):
        """
        Function to extract data. Bulk of the dataset functionality will be implemented by this method.
        :param idx: data index
        """
        inputs = self.get_inputs(idx)
        ground_truth = self.get_ground_truth(idx)
        data = {'inputs': inputs, 'ground_truth': ground_truth}
        self.save_data(idx, data)

    @abc.abstractmethod
    def get_inputs(self, idx: int) -> Dict:
        """
        Extracts model inputs.
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ground_truth(self, idx: int) -> Dict:
        """
        Extracts ground truth 'labels' for training.
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load_data(self, idx: int) -> Dict:
        """
        Function to load extracted data.
        :param idx: data index
        :return data: Dictionary with pre-processed data
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save_data(self, idx: int, data: Dict):
        """
        Function to save extracted pre-processed data.
        :param idx: data index
        :param data: Dictionary with pre-processed data
        """
        raise NotImplementedError()


class SingleAgentDataset(TrajectoryDataset):
    """
    Base class for single agent dataset. While we implicitly model all surrounding agents in the scene, predictions
    are made for a single target agent at a time.
    """

    @abc.abstractmethod
    def get_map_representation(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts map representation
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_surrounding_agent_representation(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_target_agent_representation(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts target agent representation
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_target_agent_future(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts future trajectory for target agent
        :param idx: data index
        """
        raise NotImplementedError()
