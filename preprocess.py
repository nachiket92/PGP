import argparse
from train_eval.preprocessor import preprocess_data
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)

preprocess_data(cfg, args.data_root, args.data_dir)
