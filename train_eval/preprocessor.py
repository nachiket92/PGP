from datasets.interface import TrajectoryDataset
import torch.utils.data as torch_data
from typing import List, Dict
import torch
import os
import pickle
from train_eval.initialization import get_specific_args, initialize_dataset


def preprocess_data(cfg: Dict, data_root: str, data_dir: str, compute_stats=True, extract=True):
    """
    Main function for pre-processing data

    :param cfg: Dictionary with configuration parameters
    :param data_root: Root directory for the dataset
    :param data_dir: Directory to extract pre-processed data
    :param compute_stats: Flag, whether to compute stats
    :param extract: Flag, whether to extract data
    """

    # String describing dataset type
    ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']

    # Get dataset specific args
    specific_args = get_specific_args(cfg['dataset'], data_root, cfg['version'] if 'version' in cfg.keys() else None)

    # Compute stats
    if compute_stats:
        train_set = initialize_dataset(ds_type, ['compute_stats', data_dir, cfg['train_set_args']] + specific_args)
        val_set = initialize_dataset(ds_type, ['compute_stats', data_dir, cfg['val_set_args']] + specific_args)
        test_set = initialize_dataset(ds_type, ['compute_stats', data_dir, cfg['test_set_args']] + specific_args)
        compute_dataset_stats([train_set, val_set, test_set], cfg['batch_size'], cfg['num_workers'],
                              verbose=cfg['verbosity'])

    # Extract data
    if extract:
        train_set = initialize_dataset(ds_type, ['extract_data', data_dir, cfg['train_set_args']] + specific_args)
        val_set = initialize_dataset(ds_type, ['extract_data', data_dir, cfg['val_set_args']] + specific_args)
        test_set = initialize_dataset(ds_type, ['extract_data', data_dir, cfg['test_set_args']] + specific_args)
        extract_data([train_set, val_set, test_set], cfg['batch_size'], cfg['num_workers'], verbose=cfg['verbosity'])


def compute_dataset_stats(dataset_splits: List[TrajectoryDataset], batch_size: int, num_workers: int, verbose=False):
    """
    Computes dataset stats

    :param dataset_splits: List of dataset objects usually corresponding to the train, val and test splits
    :param batch_size: Batch size for dataloader
    :param num_workers: Number of workers for dataloader
    :param verbose: Whether to print progress
    """
    # Check if all datasets have been initialized with the correct mode
    for dataset in dataset_splits:
        if dataset.mode != 'compute_stats':
            raise Exception('Dataset mode should be compute_stats')

    # Initialize data loaders
    data_loaders = []
    for dataset in dataset_splits:
        dl = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        data_loaders.append(dl)

    # Initialize dataset statistics
    stats = {}

    # For printing progress
    print("Computing dataset stats...")
    num_mini_batches = sum([len(data_loader) for data_loader in data_loaders])
    mini_batch_count = 0

    # Loop over splits and mini-batches
    for data_loader in data_loaders:
        for i, mini_batch_stats in enumerate(data_loader):
            for k, v in mini_batch_stats.items():
                if k in stats.keys():
                    stats[k] = max(stats[k], torch.max(v).item())
                else:
                    stats[k] = torch.max(v).item()

            # Show progress
            if verbose:
                print("mini batch " + str(mini_batch_count + 1) + '/' + str(num_mini_batches))
                mini_batch_count += 1

    # Save stats
    filename = os.path.join(dataset_splits[0].data_dir, 'stats.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def extract_data(dataset_splits: List[TrajectoryDataset], batch_size: int, num_workers: int, verbose=False):
    """
    Extracts pre-processed data

    :param dataset_splits: List of dataset objects usually corresponding to the train, val and test splits
    :param batch_size: Batch size for dataloader
    :param num_workers: Number of workers for dataloader
    :param verbose: Whether to print progress
    """
    # Check if all datasets have been initialized with the correct mode
    for dataset in dataset_splits:
        if dataset.mode != 'extract_data':
            raise Exception('Dataset mode should be extract_data')

    # Initialize data loaders
    data_loaders = []
    for dataset in dataset_splits:
        dl = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        data_loaders.append(dl)

    # For printing progress
    print("Extracting pre-processed data...")
    num_mini_batches = sum([len(data_loader) for data_loader in data_loaders])
    mini_batch_count = 0

    # Loop over splits and mini-batches
    for data_loader in data_loaders:
        for i, _ in enumerate(data_loader):

            # Show progress
            if verbose:
                print("mini batch " + str(mini_batch_count + 1) + '/' + str(num_mini_batches))
                mini_batch_count += 1
