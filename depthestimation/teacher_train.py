import torch

from depthestimation.trainer import Trainer
from depthestimation.options_dummy_model import DepthOptions
#from depthestimation.options_ucl import DepthOptions
#from depthestimation.options_mid import DepthOptions


options = DepthOptions()
opts = options.parse()

if __name__ == "__main__":
    train = Trainer(opts)
    train.train()
    

