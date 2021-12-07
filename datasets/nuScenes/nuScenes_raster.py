from datasets.nuScenes.nuScenes import NuScenesTrajectories
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction import PredictHelper
import numpy as np
from typing import Dict


class NuScenesRaster(NuScenesTrajectories):
    """
    NuScenes dataset class for single agent prediction, using the raster representation for maps and agents
    """

    def __init__(self, mode: str, data_dir: str, args: Dict, helper: PredictHelper):
        """
        Initialize predict helper, agent and scene representations
        :param mode: Mode of operation of dataset, one of {'compute_stats', 'extract_data', 'load_data'}
        :param data_dir: Directory to store extracted pre-processed data
        :param helper: NuScenes PredictHelper
        :param args: Dataset arguments
        """
        super().__init__(mode, data_dir, args, helper)

        # Raster parameters
        self.img_size = args['img_size']
        self.map_extent = args['map_extent']

        # Raster map with agent boxes
        resolution = (self.map_extent[1] - self.map_extent[0]) / self. img_size[1]
        self.map_rasterizer = StaticLayerRasterizer(self.helper,
                                                    resolution=resolution,
                                                    meters_ahead=self.map_extent[3],
                                                    meters_behind=-self.map_extent[2],
                                                    meters_left=-self.map_extent[0],
                                                    meters_right=self.map_extent[1])

        self.agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.t_h,
                                                           resolution=resolution,
                                                           meters_ahead=self.map_extent[3],
                                                           meters_behind=-self.map_extent[2],
                                                           meters_left=-self.map_extent[0],
                                                           meters_right=self.map_extent[1])

    def compute_stats(self, idx: int):
        """
        Function to compute dataset statistics. Nothing to compute
        """
        return {}

    def get_target_agent_representation(self, idx: int) -> np.ndarray:
        """
        Extracts target agent representation
        :param idx: data index
        :return hist: motion state for target agent, [|velocity|, |acc|, |yaw_rate|]
        """
        i_t, s_t = self.token_list[idx].split("_")

        vel = self.helper.get_velocity_for_agent(i_t, s_t)
        acc = self.helper.get_acceleration_for_agent(i_t, s_t)
        yaw_rate = self.helper.get_heading_change_rate_for_agent(i_t, s_t)

        motion_state = np.asarray([vel, acc, yaw_rate])
        for i, val in enumerate(motion_state):
            if np.isnan(val):
                motion_state[i] = 0

        return motion_state

    def get_map_representation(self, idx: int) -> np.ndarray:
        """
        Extracts map representation
        :param idx: data index
        :return img: RGB raster image with static map elements, shape: [3, img_size[0], img_size[1]]
        """
        i_t, s_t = self.token_list[idx].split("_")
        img = self.map_rasterizer.make_representation(i_t, s_t)
        img = np.moveaxis(img, -1, 0)
        img = img.astype(float) / 255
        return img

    def get_surrounding_agent_representation(self, idx: int) -> np.ndarray:
        """
        Extracts surrounding agent representation
        :param idx: data index
        :return img: Raster image with faded bounding boxes representing surrounding agents,
         shape: [3, img_size[0], img_size[1]]
        """
        i_t, s_t = self.token_list[idx].split("_")
        img = self.agent_rasterizer.make_representation(i_t, s_t)
        img = np.moveaxis(img, -1, 0)
        img = img.astype(float) / 255
        return img
