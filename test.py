import torch
import logging
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import random

import numpy as np
import torch

from src.utils import config
from src.train import train_utils
import src.models as models
from src.utils.datasets import get_dataset

class TestUtils(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = self.cfg['mode']
        self.batch_size = self.cfg['test']['batch_size']

    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.cfg['data']['input_folder'] = self.cfg['data']['test_input_folder']
        self.dataset = get_dataset(cfg=self.cfg, device=self.device)
        self.test_dataloaders = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # load model
        if self.mode == 'visible':
            self.net = getattr(models, 'CNN')(self.cfg)
            self.net.load_state_dict(torch.load('saved_models/RGB_model.pth'))
        elif self.mode == 'IR':
            self.net = getattr(models, 'IRCNN')(self.cfg)
            self.net.load_state_dict(torch.load('saved_models/IR_model.pth'))

        self.net = self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

    def test(self):
        correct = 0
        total = 0
        avg_test_loss = 0.0
        self.net.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.test_dataloaders), desc="Testing data"):
                labels = data[2].long()
                img = data[1]

                outputs = self.net(img)

                # calculate accuracy
                predicted = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # calculate loss
                avg_test_loss += self.criterion(outputs, labels) / len(self.test_dataloaders)
        print('TESTING:')
        print(f'Accuracy: {100 * correct / total:.2f} %')
        print(f'Average loss: {avg_test_loss:.3f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for running the RGB IR-Fusion.')
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    test_utils = TestUtils(cfg)
    test_utils.setup()
    test_utils.test()
