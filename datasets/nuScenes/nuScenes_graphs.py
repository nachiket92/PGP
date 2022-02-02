import matplotlib.pyplot as plt
from datasets.nuScenes.nuScenes_vector import NuScenesVector
from nuscenes.prediction.input_representation.static_layers import color_by_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
import numpy as np
from typing import Dict, Tuple, Union, List
from scipy.spatial.distance import cdist


class NuScenesGraphs(NuScenesVector):
    """
    NuScenes dataset class for single agent prediction, using the graph representation from PGP for maps and agents
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
        self.traversal_horizon = args['traversal_horizon']

        # Load dataset stats (max nodes, max agents etc.)
        if self.mode == 'extract_data':
            stats = self.load_stats()
            self.max_nbr_nodes = stats['max_nbr_nodes']

    def compute_stats(self, idx: int) -> Dict[str, int]:
        """
        Function to compute statistics for a given data point
        """
        num_lane_nodes, max_nbr_nodes = self.get_map_representation(idx)
        num_vehicles, num_pedestrians = self.get_surrounding_agent_representation(idx)
        stats = {
            'num_lane_nodes': num_lane_nodes,
            'max_nbr_nodes': max_nbr_nodes,
            'num_vehicles': num_vehicles,
            'num_pedestrians': num_pedestrians
        }

        return stats

    def extract_data(self, idx: int):
        """
        Function to extract data. Bulk of the dataset functionality will be implemented by this method.
        :param idx: data index
        """
        inputs = self.get_inputs(idx)
        ground_truth = self.get_ground_truth(idx)
        node_seq_gt, evf_gt = self.get_visited_edges(idx, inputs['map_representation'])
        init_node = self.get_initial_node(inputs['map_representation'])

        ground_truth['evf_gt'] = evf_gt
        inputs['init_node'] = init_node
        inputs['node_seq_gt'] = node_seq_gt  # For pretraining with ground truth node sequence
        data = {'inputs': inputs, 'ground_truth': ground_truth}
        self.save_data(idx, data)

    def get_inputs(self, idx: int) -> Dict:
        inputs = super().get_inputs(idx)
        a_n_masks = self.get_agent_node_masks(inputs['map_representation'], inputs['surrounding_agent_representation'])
        inputs['agent_node_masks'] = a_n_masks
        return inputs

    def get_ground_truth(self, idx: int) -> Dict:
        ground_truth = super().get_ground_truth(idx)
        return ground_truth

    def get_map_representation(self, idx: int) -> Union[Tuple[int, int], Dict]:
        """
        Extracts map representation
        :param idx: data index
        :return: Returns an ndarray with lane node features, shape [max_nodes, polyline_length, 5] and an ndarray of
            masks of the same shape, with value 1 if the nodes/poses are empty,
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_name = self.helper.get_map_name_from_sample_token(s_t)
        map_api = self.maps[map_name]

        # Get agent representation in global co-ordinates
        global_pose = self.get_target_agent_global_pose(idx)

        # Get lanes around agent within map_extent
        lanes = self.get_lanes_around_agent(global_pose, map_api)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(global_pose, map_api)

        # Get vectorized representation of lanes
        lane_node_feats, lane_ids = self.get_lane_node_feats(global_pose, lanes, polygons)

        # Discard lanes outside map extent
        lane_node_feats, lane_ids = self.discard_poses_outside_extent(lane_node_feats, lane_ids)

        # Get edges:
        e_succ = self.get_successor_edges(lane_ids, map_api)
        e_prox = self.get_proximal_edges(lane_node_feats, e_succ)

        # Concatentate flag indicating whether a node hassss successors to lane node feats
        lane_node_feats = self.add_boundary_flag(e_succ, lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, 6))]
            e_succ = [[]]
            e_prox = [[]]

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == 'compute_stats':

            num_nbrs = [len(e_succ[i]) + len(e_prox[i]) for i in range(len(e_succ))]
            max_nbrs = max(num_nbrs) if len(num_nbrs) > 0 else 0
            num_nodes = len(lane_node_feats)

            return num_nodes, max_nbrs

        # Get edge lookup tables
        s_next, edge_type = self.get_edge_lookup(e_succ, e_prox)

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, 6)

        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
            's_next': s_next,
            'edge_type': edge_type
        }

        return map_representation

    @staticmethod
    def get_successor_edges(lane_ids: List[str], map_api: NuScenesMap) -> List[List[int]]:
        """
        Returns successor edge list for each node
        """
        e_succ = []
        for node_id, lane_id in enumerate(lane_ids):
            e_succ_node = []
            if node_id + 1 < len(lane_ids) and lane_id == lane_ids[node_id + 1]:
                e_succ_node.append(node_id + 1)
            else:
                outgoing_lane_ids = map_api.get_outgoing_lane_ids(lane_id)
                for outgoing_id in outgoing_lane_ids:
                    if outgoing_id in lane_ids:
                        e_succ_node.append(lane_ids.index(outgoing_id))

            e_succ.append(e_succ_node)

        return e_succ

    @staticmethod
    def get_proximal_edges(lane_node_feats: List[np.ndarray], e_succ: List[List[int]],
                           dist_thresh=4, yaw_thresh=np.pi/4) -> List[List[int]]:
        """
        Returns proximal edge list for each node
        """
        e_prox = [[] for _ in lane_node_feats]
        for src_node_id, src_node_feats in enumerate(lane_node_feats):
            for dest_node_id in range(src_node_id + 1, len(lane_node_feats)):
                if dest_node_id not in e_succ[src_node_id] and src_node_id not in e_succ[dest_node_id]:
                    dest_node_feats = lane_node_feats[dest_node_id]
                    pairwise_dist = cdist(src_node_feats[:, :2], dest_node_feats[:, :2])
                    min_dist = np.min(pairwise_dist)
                    if min_dist <= dist_thresh:
                        yaw_src = np.arctan2(np.mean(np.sin(src_node_feats[:, 2])),
                                             np.mean(np.cos(src_node_feats[:, 2])))
                        yaw_dest = np.arctan2(np.mean(np.sin(dest_node_feats[:, 2])),
                                              np.mean(np.cos(dest_node_feats[:, 2])))
                        yaw_diff = np.arctan2(np.sin(yaw_src-yaw_dest), np.cos(yaw_src-yaw_dest))
                        if np.absolute(yaw_diff) <= yaw_thresh:
                            e_prox[src_node_id].append(dest_node_id)
                            e_prox[dest_node_id].append(src_node_id)

        return e_prox

    @staticmethod
    def add_boundary_flag(e_succ: List[List[int]], lane_node_feats: np.ndarray):
        """
        Adds a binary flag to lane node features indicating whether the lane node has any successors.
        Serves as an indicator for boundary nodes.
        """
        for n, lane_node_feat_array in enumerate(lane_node_feats):
            flag = 1 if len(e_succ[n]) == 0 else 0
            lane_node_feats[n] = np.concatenate((lane_node_feat_array, flag * np.ones((len(lane_node_feat_array), 1))),
                                                axis=1)

        return lane_node_feats

    def get_edge_lookup(self, e_succ: List[List[int]], e_prox: List[List[int]]):
        """
        Returns edge look up tables
        :param e_succ: Lists of successor edges for each node
        :param e_prox: Lists of proximal edges for each node
        :return:

        s_next: Look-up table mapping source node to destination node for each edge. Each row corresponds to
        a source node, with entries corresponding to destination nodes. Last entry is always a terminal edge to a goal
        state at that node. shape: [max_nodes, max_nbr_nodes + 1]. Last

        edge_type: Look-up table of the same shape as s_next containing integer values for edge types.
        {0: No edge exists, 1: successor edge, 2: proximal edge, 3: terminal edge}
        """

        s_next = np.zeros((self.max_nodes, self.max_nbr_nodes + 1))
        edge_type = np.zeros((self.max_nodes, self.max_nbr_nodes + 1), dtype=int)

        for src_node in range(len(e_succ)):
            nbr_idx = 0
            successors = e_succ[src_node]
            prox_nodes = e_prox[src_node]

            # Populate successor edges
            for successor in successors:
                s_next[src_node, nbr_idx] = successor
                edge_type[src_node, nbr_idx] = 1
                nbr_idx += 1

            # Populate proximal edges
            for prox_node in prox_nodes:
                s_next[src_node, nbr_idx] = prox_node
                edge_type[src_node, nbr_idx] = 2
                nbr_idx += 1

            # Populate terminal edge
            s_next[src_node, -1] = src_node + self.max_nodes
            edge_type[src_node, -1] = 3

        return s_next, edge_type

    def get_initial_node(self, lane_graph: Dict) -> np.ndarray:
        """
        Returns initial node probabilities for initializing the graph traversal policy
        :param lane_graph: lane graph dictionary with lane node features and edge look-up tables
        """

        # Unpack lane node poses
        node_feats = lane_graph['lane_node_feats']
        node_feat_lens = np.sum(1 - lane_graph['lane_node_masks'][:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[:int(node_feat_lens[i]), :3])

        assigned_nodes = self.assign_pose_to_node(node_poses, np.asarray([0, 0, 0]), dist_thresh=3,
                                                  yaw_thresh=np.pi / 4, return_multiple=True)

        init_node = np.zeros(self.max_nodes)
        init_node[assigned_nodes] = 1/len(assigned_nodes)
        return init_node

    def get_visited_edges(self, idx: int, lane_graph: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns nodes and edges of the lane graph visited by the actual target vehicle in the future. This serves as
        ground truth for training the graph traversal policy pi_route.

        :param idx: dataset index
        :param lane_graph: lane graph dictionary with lane node features and edge look-up tables
        :return: node_seq: Sequence of visited node ids.
                 evf: Look-up table of visited edges.
        """

        # Unpack lane graph dictionary
        node_feats = lane_graph['lane_node_feats']
        s_next = lane_graph['s_next']
        edge_type = lane_graph['edge_type']

        node_feat_lens = np.sum(1 - lane_graph['lane_node_masks'][:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[:int(node_feat_lens[i]), :3])

        # Initialize outputs
        current_step = 0
        node_seq = np.zeros(self.traversal_horizon)
        evf = np.zeros_like(s_next)

        # Get future trajectory:
        i_t, s_t = self.token_list[idx].split("_")
        fut_xy = self.helper.get_future_for_agent(i_t, s_t, 6, True)
        fut_interpolated = np.zeros((fut_xy.shape[0] * 10 + 1, 2))
        param_query = np.linspace(0, fut_xy.shape[0], fut_xy.shape[0] * 10 + 1)
        param_given = np.linspace(0, fut_xy.shape[0], fut_xy.shape[0] + 1)
        val_given_x = np.concatenate(([0], fut_xy[:, 0]))
        val_given_y = np.concatenate(([0], fut_xy[:, 1]))
        fut_interpolated[:, 0] = np.interp(param_query, param_given, val_given_x)
        fut_interpolated[:, 1] = np.interp(param_query, param_given, val_given_y)
        fut_xy = fut_interpolated

        # Compute yaw values for future:
        fut_yaw = np.zeros(len(fut_xy))
        for n in range(1, len(fut_yaw)):
            fut_yaw[n] = -np.arctan2(fut_xy[n, 0] - fut_xy[n-1, 0], fut_xy[n, 1] - fut_xy[n-1, 1])

        # Loop over future trajectory poses
        query_pose = np.asarray([fut_xy[0, 0], fut_xy[0, 1], fut_yaw[0]])
        current_node = self.assign_pose_to_node(node_poses, query_pose)
        node_seq[current_step] = current_node
        for n in range(1, len(fut_xy)):
            query_pose = np.asarray([fut_xy[n, 0], fut_xy[n, 1], fut_yaw[n]])
            dist_from_current_node = np.min(np.linalg.norm(node_poses[current_node][:, :2] - query_pose[:2], axis=1))

            # If pose has deviated sufficiently from current node and is within area of interest, assign to a new node
            padding = self.polyline_length * self.polyline_resolution / 2
            if self.map_extent[0] - padding <= query_pose[0] <= self.map_extent[1] + padding and \
                    self.map_extent[2] - padding <= query_pose[1] <= self.map_extent[3] + padding:

                if dist_from_current_node >= 1.5:
                    assigned_node = self.assign_pose_to_node(node_poses, query_pose)

                    # Assign new node to node sequence and edge to visited edges
                    if assigned_node != current_node:

                        if assigned_node in s_next[current_node]:
                            nbr_idx = np.where(s_next[current_node] == assigned_node)[0]
                            nbr_valid = np.where(edge_type[current_node] > 0)[0]
                            nbr_idx = np.intersect1d(nbr_idx, nbr_valid)

                            if edge_type[current_node, nbr_idx] > 0:
                                evf[current_node, nbr_idx] = 1

                        current_node = assigned_node
                        if current_step < self.traversal_horizon-1:
                            current_step += 1
                            node_seq[current_step] = current_node

            else:
                break

        # Assign goal node and edge
        goal_node = current_node + self.max_nodes
        node_seq[current_step + 1:] = goal_node
        evf[current_node, -1] = 1

        return node_seq, evf

    @staticmethod
    def assign_pose_to_node(node_poses, query_pose, dist_thresh=5, yaw_thresh=np.pi/3, return_multiple=False):
        """
        Assigns a given agent pose to a lane node. Takes into account distance from the lane centerline as well as
        direction of motion.
        """
        dist_vals = []
        yaw_diffs = []

        for i in range(len(node_poses)):
            distances = np.linalg.norm(node_poses[i][:, :2] - query_pose[:2], axis=1)
            dist_vals.append(np.min(distances))
            idx = np.argmin(distances)
            yaw_lane = node_poses[i][idx, 2]
            yaw_query = query_pose[2]
            yaw_diffs.append(np.arctan2(np.sin(yaw_lane - yaw_query), np.cos(yaw_lane - yaw_query)))

        idcs_yaw = np.where(np.absolute(np.asarray(yaw_diffs)) <= yaw_thresh)[0]
        idcs_dist = np.where(np.asarray(dist_vals) <= dist_thresh)[0]
        idcs = np.intersect1d(idcs_dist, idcs_yaw)

        if len(idcs) > 0:
            if return_multiple:
                return idcs
            assigned_node_id = idcs[int(np.argmin(np.asarray(dist_vals)[idcs]))]
        else:
            assigned_node_id = np.argmin(np.asarray(dist_vals))
            if return_multiple:
                assigned_node_id = np.asarray([assigned_node_id])

        return assigned_node_id

    @staticmethod
    def get_agent_node_masks(hd_map: Dict, agents: Dict, dist_thresh=10) -> Dict:
        """
        Returns key/val masks for agent-node attention layers. All agents except those within a distance threshold of
        the lane node are masked. The idea is to incorporate local agent context at each lane node.
        """

        lane_node_feats = hd_map['lane_node_feats']
        lane_node_masks = hd_map['lane_node_masks']
        vehicle_feats = agents['vehicles']
        vehicle_masks = agents['vehicle_masks']
        ped_feats = agents['pedestrians']
        ped_masks = agents['pedestrian_masks']

        vehicle_node_masks = np.ones((len(lane_node_feats), len(vehicle_feats)))
        ped_node_masks = np.ones((len(lane_node_feats), len(ped_feats)))

        for i, node_feat in enumerate(lane_node_feats):
            if (lane_node_masks[i] == 0).any():
                node_pose_idcs = np.where(lane_node_masks[i][:, 0] == 0)[0]
                node_locs = node_feat[node_pose_idcs, :2]

                for j, vehicle_feat in enumerate(vehicle_feats):
                    if (vehicle_masks[j] == 0).any():
                        vehicle_loc = vehicle_feat[-1, :2]
                        dist = np.min(np.linalg.norm(node_locs - vehicle_loc, axis=1))
                        if dist <= dist_thresh:
                            vehicle_node_masks[i, j] = 0

                for j, ped_feat in enumerate(ped_feats):
                    if (ped_masks[j] == 0).any():
                        ped_loc = ped_feat[-1, :2]
                        dist = np.min(np.linalg.norm(node_locs - ped_loc, axis=1))
                        if dist <= dist_thresh:
                            ped_node_masks[i, j] = 0

        agent_node_masks = {'vehicles': vehicle_node_masks, 'pedestrians': ped_node_masks}
        return agent_node_masks

    def visualize_graph(self, node_feats, s_next, edge_type, evf_gt, node_seq, fut_xy):
        """
        Function to visualize lane graph.
        """
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((3, 3)), extent=self.map_extent, cmap='gist_gray')

        # Plot edges
        for src_id, src_feats in enumerate(node_feats):
            feat_len = np.sum(np.sum(np.absolute(src_feats), axis=1) != 0)

            if feat_len > 0:
                src_x = np.mean(src_feats[:feat_len, 0])
                src_y = np.mean(src_feats[:feat_len, 1])

                for idx, dest_id in enumerate(s_next[src_id]):
                    edge_t = edge_type[src_id, idx]
                    visited = evf_gt[src_id, idx]
                    if 3 > edge_t > 0:

                        dest_feats = node_feats[int(dest_id)]
                        feat_len_dest = np.sum(np.sum(np.absolute(dest_feats), axis=1) != 0)
                        dest_x = np.mean(dest_feats[:feat_len_dest, 0])
                        dest_y = np.mean(dest_feats[:feat_len_dest, 1])
                        d_x = dest_x - src_x
                        d_y = dest_y - src_y

                        line_style = '-' if edge_t == 1 else '--'
                        width = 2 if visited else 0.01
                        alpha = 1 if visited else 0.5

                        plt.arrow(src_x, src_y, d_x, d_y, color='w', head_width=0.1, length_includes_head=True,
                                  linestyle=line_style, width=width, alpha=alpha)

        # Plot nodes
        for node_id, node_feat in enumerate(node_feats):
            feat_len = np.sum(np.sum(np.absolute(node_feat), axis=1) != 0)
            if feat_len > 0:
                visited = node_id in node_seq
                x = np.mean(node_feat[:feat_len, 0])
                y = np.mean(node_feat[:feat_len, 1])
                yaw = np.arctan2(np.mean(np.sin(node_feat[:feat_len, 2])),
                                 np.mean(np.cos(node_feat[:feat_len, 2])))
                c = color_by_yaw(0, yaw)
                c = np.asarray(c).reshape(-1, 3) / 255
                s = 200 if visited else 50
                ax.scatter(x, y, s, c=c)

        plt.plot(fut_xy[:, 0], fut_xy[:, 1], color='r', lw=3)

        plt.show()
