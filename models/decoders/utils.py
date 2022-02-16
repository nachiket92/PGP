import torch
from datasets.interface import SingleAgentDataset
import numpy as np
from sklearn.cluster import KMeans
import psutil
import ray
from scipy.spatial.distance import cdist


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize ray:
num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus, log_to_driver=False)


def k_means_anchors(k, ds: SingleAgentDataset):
    """
    Extracts anchors for multipath/covernet using k-means on train set trajectories
    """
    prototype_traj = ds[0]['ground_truth']['traj']
    traj_len = prototype_traj.shape[0]
    traj_dim = prototype_traj.shape[1]
    ds_size = len(ds)
    trajectories = np.zeros((ds_size, traj_len, traj_dim))
    for i, data in enumerate(ds):
        trajectories[i] = data['ground_truth']['traj']
    clustering = KMeans(n_clusters=k).fit(trajectories.reshape((ds_size, -1)))
    anchors = np.zeros((k, traj_len, traj_dim))
    for i in range(k):
        anchors[i] = np.mean(trajectories[clustering.labels_ == i], axis=0)
    anchors = torch.from_numpy(anchors).float().to(device)
    return anchors


def bivariate_gaussian_activation(ip: torch.Tensor) -> torch.Tensor:
    """
    Activation function to output parameters of bivariate Gaussian distribution
    """
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


@ray.remote
def cluster_and_rank(k: int, data: np.ndarray):
    """
    Combines the clustering and ranking steps so that ray.remote gets called just once
    """

    def cluster(n_clusters: int, x: np.ndarray):
        """
        Cluster using Scikit learn
        """
        clustering_op = KMeans(n_clusters=n_clusters, n_init=1, max_iter=100, init='random').fit(x)
        return clustering_op.labels_, clustering_op.cluster_centers_

    def rank_clusters(cluster_counts, cluster_centers):
        """
        Rank the K clustered trajectories using Ward's criterion. Start with K cluster centers and cluster counts.
        Find the two clusters to merge based on Ward's criterion. Smaller of the two will get assigned rank K.
        Merge the two clusters. Repeat process to assign ranks K-1, K-2, ..., 2.
        """

        num_clusters = len(cluster_counts)
        cluster_ids = np.arange(num_clusters)
        ranks = np.ones(num_clusters)

        for i in range(num_clusters, 0, -1):
            # Compute Ward distances:
            centroid_dists = cdist(cluster_centers, cluster_centers)
            n1 = cluster_counts.reshape(1, -1).repeat(len(cluster_counts), axis=0)
            n2 = n1.transpose()
            wts = n1 * n2 / (n1 + n2)
            dists = wts * centroid_dists + np.diag(np.inf * np.ones(len(cluster_counts)))

            # Get clusters with min Ward distance and select cluster with fewer counts
            c1, c2 = np.unravel_index(dists.argmin(), dists.shape)
            c = c1 if cluster_counts[c1] <= cluster_counts[c2] else c2
            c_ = c2 if cluster_counts[c1] <= cluster_counts[c2] else c1

            # Assign rank i to selected cluster
            ranks[cluster_ids[c]] = i

            # Merge clusters and update identity of merged cluster
            cluster_centers[c_] = (cluster_counts[c_] * cluster_centers[c_] + cluster_counts[c] * cluster_centers[c]) /\
                                  (cluster_counts[c_] + cluster_counts[c])
            cluster_counts[c_] += cluster_counts[c]

            # Discard merged cluster
            cluster_ids = np.delete(cluster_ids, c)
            cluster_centers = np.delete(cluster_centers, c, axis=0)
            cluster_counts = np.delete(cluster_counts, c)

        return ranks

    cluster_lbls, cluster_ctrs = cluster(k, data)
    cluster_cnts = np.unique(cluster_lbls, return_counts=True)[1]
    cluster_ranks = rank_clusters(cluster_cnts.copy(), cluster_ctrs.copy())
    return {'lbls': cluster_lbls, 'ranks': cluster_ranks, 'counts': cluster_cnts}


def cluster_traj(k: int, traj: torch.Tensor):
    """
    clusters sampled trajectories to output K modes.
    :param k: number of clusters
    :param traj: set of sampled trajectories, shape [batch_size, num_samples, traj_len, 2]
    :return: traj_clustered:  set of clustered trajectories, shape [batch_size, k, traj_len, 2]
             scores: scores for clustered trajectories (basically 1/rank), shape [batch_size, k]
    """

    # Initialize output tensors
    batch_size = traj.shape[0]
    num_samples = traj.shape[1]
    traj_len = traj.shape[2]

    # Down-sample traj along time dimension for faster clustering
    data = traj[:, :, 0::3, :]
    data = data.reshape(batch_size, num_samples, -1).detach().cpu().numpy()

    # Cluster and rank
    cluster_ops = ray.get([cluster_and_rank.remote(k, data_slice) for data_slice in data])
    cluster_lbls = [cluster_op['lbls'] for cluster_op in cluster_ops]
    cluster_counts = [cluster_op['counts'] for cluster_op in cluster_ops]
    cluster_ranks = [cluster_op['ranks'] for cluster_op in cluster_ops]

    # Compute mean (clustered) traj and scores
    lbls = torch.as_tensor(cluster_lbls, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2).long()
    traj_summed = torch.zeros(batch_size, k, traj_len, 2, device=device).scatter_add(1, lbls, traj)
    cnt_tensor = torch.as_tensor(cluster_counts, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2)
    traj_clustered = traj_summed / cnt_tensor
    scores = 1 / torch.as_tensor(cluster_ranks, device=device)
    scores = scores / torch.sum(scores, dim=1)[0]

    return traj_clustered, scores
