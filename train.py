############
## IMPORT ##
############

import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader    import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from models.discriminator           import Discriminator
from models.generator               import Generator
from torch.optim                    import Adam
from torch.optim.lr_scheduler       import StepLR

###########################
## CONSTANTS AND GLOBALS ##
###########################

DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} to train the model.".format(DEVICE))

###########################
## Classes and functions ##
###########################

##########
## Main ##
##########

if __name__ == '__main__':

    print("Classifier training loop.")


    if os.getcwd()[-8:] != 'dc_gan':
        message = 'Run the file from the the root:\n'
        message += 'cd dc_gan\n'
        message += 'python train.py'
        raise Exception(message)

    # Load params
    print("Loading params")
    print("Params loaded.")

    # Load the data
    print("Loading dataset")
    print("Dataset loaded.")

    # Load the model
    print("Loading the model")
    print("Model loaded.")

    # Define optimizer and loss function
    print("Selecting optimizer and loss function")
    optimizer = Adam()
    print("Done.")

    # Training loop
