import torch
import argparse

device = 'gpu' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description="Get basic configuration")

    parser.add_argument('--project_name',
                        type=str,
                        default='test',
                        help="Project name used for Weights & Biases monitoring.")

    # Data
    parser.add_argument('--dataset',
                        type=str,
                        # default='CelebA',
                        default='CUB-200-2011',
                        # default='MNIST',
                        # default='XOR',
                        # default='Dot',
                        # default='Trigonometric',
                        # default='vector',
                        help='Dataset name')

    # Operation environment
    parser.add_argument('--seed',
                        type=int,
                        default=2002,
                        help='Random seed')
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='Running on which device')

    # Training hyper-parameters
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='Batch size')  # 1024
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=32,
                        help='Gradient accumulation to solve the GPU memory problem')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--patience',
                        type=int,
                        default=4,
                        help='the patience for early stopping')

    # Experiment configuration
    parser.add_argument('--port',
                        type=int,
                        default=19923,
                        help='Python console use only')
    parser.add_argument('--save_path',
                        type=str,
                        default='./checkpoints/',
                        help='Checkpoints saving path')

    parser.add_argument('--activation_freq',
                        type=int,
                        default=0,
                        help='How frequently in terms of epochs should we store the embedding activations')
    parser.add_argument('--single_frequency_epochs',
                        type=int,
                        default=0,
                        help='Store the embedding every epoch or not')

    args = parser.parse_args()
    return args
