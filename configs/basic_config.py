import torch
import argparse

device = 'gpu' if torch.cuda.is_available() else 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description="Get basic configuration")

    parser.add_argument(
        '--project_name',
        type=str,
        default='test',
        help="Project name used for Weights & Biases monitoring.")

    # Data
    parser.add_argument(
        '--dataset',
        type=str,
        # default='CelebA',
        default='CUB-200-2011',
        # default='PBC',
        # default='7pt',
        # default='AwA2',
        help='Dataset name')
    parser.add_argument(
        '--labeled_ratio',
        type=float,
        default=0.1,
        help='The proportion of the labeled data')
    parser.add_argument(
        '--image_encoder',
        type=str,
        # default='resnet18',
        default='resnet34',
        # default='resnet50',
        # default='densenet121',
        help='Dataset name')

    # Operation environment
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed')
    parser.add_argument(
        '--device',
        type=str,
        default=device,
        help='Running on which device')

    # Experiment configuration
    parser.add_argument(
        '--port',
        type=int,
        default=19923,
        help='Python console use only')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./checkpoints/',
        help='Checkpoints saving path')

    parser.add_argument(
        '--activation_freq',
        type=int,
        default=0,
        help='How frequently in terms of epochs should we store the embedding activations')
    parser.add_argument(
        '--single_frequency_epochs',
        type=int,
        default=0,
        help='Store the embedding every epoch or not')

    args = parser.parse_args()
    return args
