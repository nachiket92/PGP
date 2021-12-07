import argparse
import yaml
from train_eval.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
parser.add_argument("-o", "--output_dir", help="Directory to save checkpoints and logs", required=True)
parser.add_argument("-n", "--num_epochs", help="Number of epochs to run training for", required=True)
parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=False)
args = parser.parse_args()


# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
    os.mkdir(os.path.join(args.output_dir, 'checkpoints'))
if not os.path.isdir(os.path.join(args.output_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(args.output_dir, 'tensorboard_logs'))


# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)


# Initialize tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))


# Train
trainer = Trainer(cfg, args.data_root, args.data_dir, checkpoint_path=args.checkpoint, writer=writer)
trainer.train(num_epochs=int(args.num_epochs), output_dir=args.output_dir)


# Close tensorboard writer
writer.close()
