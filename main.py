from __future__ import print_function
import argparse
from torch.utils.data import DataLoader
from train_test import SRCNNTrainer

from data import get_training_set, get_test_set
from option import args


def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    train_set = get_training_set(args.upscale_factor)
    test_set = get_test_set(args.upscale_factor)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

    model = SRCNNTrainer(args, training_data_loader, testing_data_loader)

    model.run()


if __name__ == '__main__':
    main()
