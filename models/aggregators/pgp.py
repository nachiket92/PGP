import torch
import torch.nn as nn
from models.aggregators.aggregator import PredictionAggregator
from typing import Dict
from torch.distributions import Categorical
from positional_encodings import PositionalEncoding1D


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PGP(PredictionAggregator):
    """
    Policy header + selective aggregator from "Multimodal trajectory prediction conditioned on lane graph traversals"
    1) Outputs edge probabilities corresponding to pi_route
    2) Samples pi_route to output traversed paths
    3) Selectively aggregates context along traversed paths
    """

    def __init__(self, args: Dict):
        """
        args to include
        'pre_train': bool, whether the model is being pre-trained using ground truth node sequence.
        'node_enc_size': int, size of node encoding
        'target_agent_enc_size': int, size of target agent encoding
        'pi_h1_size': int, size of first layer of policy header
        'pi_h2_size': int, size of second layer of policy header
        'emb_size': int, embedding size for attention layer for aggregating node encodings
        'num_heads: int, number of attention heads
        'num_samples': int, number of sampled traversals (and encodings) to output
        """

        super().__init__()
        self.pre_train = args['pre_train']

        # Policy header
        self.pi_h1 = nn.Linear(2 * args['node_enc_size'] + args['target_agent_enc_size'] + 2, args['pi_h1_size'])
        self.pi_h2 = nn.Linear(args['pi_h1_size'], args['pi_h2_size'])
        self.pi_op = nn.Linear(args['pi_h2_size'], 1)
        self.pi_h1_goal = nn.Linear(args['node_enc_size'] + args['target_agent_enc_size'], args['pi_h1_size'])
        self.pi_h2_goal = nn.Linear(args['pi_h1_size'], args['pi_h2_size'])
        self.pi_op_goal = nn.Linear(args['pi_h2_size'], 1)
        self.leaky_relu = nn.LeakyReLU()
        self.log_softmax = nn.LogSoftmax(dim=2)

        # For sampling policy
        self.horizon = args['horizon']
        self.num_samples = args['num_samples']

        # Attention based aggregator
        self.pos_enc = PositionalEncoding1D(args['node_enc_size'])
        self.query_emb = nn.Linear(args['target_agent_enc_size'], args['emb_size'])
        self.key_emb = nn.Linear(args['node_enc_size'], args['emb_size'])
        self.val_emb = nn.Linear(args['node_enc_size'], args['emb_size'])
        self.mha = nn.MultiheadAttention(args['emb_size'], args['num_heads'])

    def forward(self, encodings: Dict) -> Dict:
        """
        Forward pass for PGP aggregator
        :param encodings: dictionary with encoder outputs
        :return: outputs: dictionary with
            'agg_encoding': aggregated encodings along sampled traversals
            'pi': discrete policy (probabilities over outgoing edges) for graph traversal
        """

        # Unpack encodings:
        target_agent_encoding = encodings['target_agent_encoding']
        node_encodings = encodings['context_encoding']['combined']
        node_masks = encodings['context_encoding']['combined_masks']
        s_next = encodings['s_next']
        edge_type = encodings['edge_type']

        # Compute pi (log probs)
        pi = self.compute_policy(target_agent_encoding, node_encodings, node_masks, s_next, edge_type)

        # If pretraining model, use ground truth node sequences
        if self.pre_train and self.training:
            sampled_traversals = encodings['node_seq_gt'].unsqueeze(1).repeat(1, self.num_samples, 1).long()
        else:
            # Sample pi
            init_node = encodings['init_node']
            sampled_traversals = self.sample_policy(torch.exp(pi), s_next, init_node)

        # Selectively aggregate context along traversed paths
        agg_enc = self.aggregate(sampled_traversals, node_encodings, target_agent_encoding)

        outputs = {'agg_encoding': agg_enc, 'pi': pi}
        return outputs

    def aggregate(self, sampled_traversals, node_encodings, target_agent_encoding) -> torch.Tensor:

        # Useful variables:
        batch_size = node_encodings.shape[0]
        max_nodes = node_encodings.shape[1]

        # Get unique traversals and form consolidated batch:
        unique_traversals = [torch.unique(i, dim=0, return_counts=True) for i in sampled_traversals]
        traversals_batched = torch.cat([i[0] for i in unique_traversals], dim=0)
        counts_batched = torch.cat([i[1] for i in unique_traversals], dim=0)
        batch_idcs = torch.cat([n*torch.ones(len(i[1])).long() for n, i in enumerate(unique_traversals)])
        batch_idcs = batch_idcs.unsqueeze(1).repeat(1, self.horizon)

        # Dummy encodings for goal nodes
        dummy_enc = torch.zeros_like(node_encodings)
        node_encodings = torch.cat((node_encodings, dummy_enc), dim=1)

        # Gather node encodings along traversed paths
        node_enc_selected = node_encodings[batch_idcs, traversals_batched]

        # Add positional encodings:
        pos_enc = self.pos_enc(torch.zeros_like(node_enc_selected))
        node_enc_selected += pos_enc

        # Multi-head attention
        target_agent_enc_batched = target_agent_encoding[batch_idcs[:, 0]]
        query = self.query_emb(target_agent_enc_batched).unsqueeze(0)
        keys = self.key_emb(node_enc_selected).permute(1, 0, 2)
        vals = self.val_emb(node_enc_selected).permute(1, 0, 2)
        key_padding_mask = torch.as_tensor(traversals_batched >= max_nodes)
        att_op, _ = self.mha(query, keys, vals, key_padding_mask)

        # Repeat based on counts
        att_op = att_op.squeeze(0).repeat_interleave(counts_batched, dim=0).view(batch_size, self.num_samples, -1)

        # Concatenate target agent encoding
        agg_enc = torch.cat((target_agent_encoding.unsqueeze(1).repeat(1, self.num_samples, 1), att_op), dim=-1)

        return agg_enc

    def sample_policy(self, pi, s_next, init_node) -> torch.Tensor:
        """
        Sample graph traversals using discrete policy.
        :param pi: tensor with probabilities corresponding to the policy
        :param s_next: look-up table for next node for a given source node and edge
        :param init_node: initial node to start the policy at
        :return:
        """
        with torch.no_grad():

            # Useful variables:
            batch_size = pi.shape[0]
            max_nodes = pi.shape[1]
            batch_idcs = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, self.num_samples).view(-1)

            # Initialize output
            sampled_traversals = torch.zeros(batch_size, self.num_samples, self.horizon, device=device).long()

            # Set up dummy self transitions for goal states:
            pi_dummy = torch.zeros_like(pi)
            pi_dummy[:, :, -1] = 1
            s_next_dummy = torch.zeros_like(s_next)
            s_next_dummy[:, :, -1] = max_nodes + torch.arange(max_nodes).unsqueeze(0).repeat(batch_size, 1)
            pi = torch.cat((pi, pi_dummy), dim=1)
            s_next = torch.cat((s_next, s_next_dummy), dim=1)

            # Sample initial node:
            pi_s = init_node.unsqueeze(1).repeat(1, self.num_samples, 1).view(-1, max_nodes)
            s = Categorical(pi_s).sample()
            sampled_traversals[:, :, 0] = s.reshape(batch_size, self.num_samples)

            # Sample traversed paths for a fixed horizon
            for n in range(1, self.horizon):

                # Gather policy at appropriate indices:
                pi_s = pi[batch_idcs, s]

                # Sample edges
                a = Categorical(pi_s).sample()

                # Look-up next node
                s = s_next[batch_idcs, s, a].long()

                # Add node indices to sampled traversals
                sampled_traversals[:, :, n] = s.reshape(batch_size, self.num_samples)

        return sampled_traversals

    def compute_policy(self, target_agent_encoding, node_encodings, node_masks, s_next, edge_type) -> torch.Tensor:
        """
        Forward pass for policy header
        :param target_agent_encoding: tensor encoding the target agent's past motion
        :param node_encodings: tensor of node encodings provided by the encoder
        :param node_masks: masks indicating whether a node exists for a given index in the tensor
        :param s_next: look-up table for next node for a given source node and edge
        :param edge_type: look-up table with edge types
        :return pi: tensor with probabilities corresponding to the policy
        """
        # Useful variables:
        batch_size = node_encodings.shape[0]
        max_nodes = node_encodings.shape[1]
        max_nbrs = s_next.shape[2] - 1
        node_enc_size = node_encodings.shape[2]
        target_agent_enc_size = target_agent_encoding.shape[1]

        # Gather source node encodigns, destination node encodings, edge encodings and target agent encodings.
        src_node_enc = node_encodings.unsqueeze(2).repeat(1, 1, max_nbrs, 1)
        dst_idcs = s_next[:, :, :-1].reshape(batch_size, -1).long()
        batch_idcs = torch.arange(batch_size).unsqueeze(1).repeat(1, max_nodes * max_nbrs)
        dst_node_enc = node_encodings[batch_idcs, dst_idcs].reshape(batch_size, max_nodes, max_nbrs, node_enc_size)
        target_agent_enc = target_agent_encoding.unsqueeze(1).unsqueeze(2).repeat(1, max_nodes, max_nbrs, 1)
        edge_enc = torch.cat((torch.as_tensor(edge_type[:, :, :-1] == 1, device=device).unsqueeze(3).float(),
                              torch.as_tensor(edge_type[:, :, :-1] == 2, device=device).unsqueeze(3).float()), dim=3)
        enc = torch.cat((target_agent_enc, src_node_enc, dst_node_enc, edge_enc), dim=3)
        enc_goal = torch.cat((target_agent_enc[:, :, 0, :], src_node_enc[:, :, 0, :]), dim=2)

        # Form a single batch of encodings
        masks = torch.sum(edge_enc, dim=3, keepdim=True).bool()
        masks_goal = ~node_masks.unsqueeze(-1).bool()
        enc_batched = torch.masked_select(enc, masks).reshape(-1, target_agent_enc_size + 2*node_enc_size + 2)
        enc_goal_batched = torch.masked_select(enc_goal, masks_goal).reshape(-1, target_agent_enc_size + node_enc_size)

        # Compute scores for pi_route
        pi_ = self.pi_op(self.leaky_relu(self.pi_h2(self.leaky_relu(self.pi_h1(enc_batched)))))
        pi = torch.zeros_like(masks).float()
        pi = pi.masked_scatter_(masks, pi_).squeeze(-1)
        pi_goal_ = self.pi_op_goal(self.leaky_relu(self.pi_h2_goal(self.leaky_relu(self.pi_h1_goal(enc_goal_batched)))))
        pi_goal = torch.zeros_like(masks_goal).float()
        pi_goal = pi_goal.masked_scatter_(masks_goal, pi_goal_)

        # Normalize to give log probabilities
        pi = torch.cat((pi, pi_goal), dim=-1)
        op_masks = torch.log(torch.as_tensor(edge_type != 0).float())
        pi = self.log_softmax(pi + op_masks)

        return pi
