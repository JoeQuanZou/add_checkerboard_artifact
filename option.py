import argparse

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Checkboard Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='drcn', help='choose which model is going to use')
parser.add_argument('--train_data_dir', type=str, default='~', help='directory of training data')
parser.add_argument('--test_data_dir', type=str, default='~', help='directory of test data')

args = parser.parse_args()