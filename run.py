import argparse
import random

import numpy as np
import torch

from src.utils import config
from src.train import train_utils
from src.test import TestUtils



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(42)

    parser = argparse.ArgumentParser(description='Arguments for running the RGB IR-Fusion.')
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)

    trainer = train_utils(cfg)
    trainer.setup()
    # trainer.train() # train model

    tester = TestUtils(cfg, trainer)
    tester.setup()


    tester.EC_validate_setup()
    tester.EC_classifier()
    tester.validate_test()
    tester.EC_classifier_validate() #自己数据集
    tester.merge_feature()


if __name__ == '__main__':
    main()


    #  train CNN
    # online : train test
    # 确定 epsilon ball radius,
    # online train 分成了 train 和 test

    # knn: 1,2,3,4
    # epsilon ball: 1,2,6,7
    # intersection: 1,2 -> which region

    # 1234 ->
    # 1267 ->

