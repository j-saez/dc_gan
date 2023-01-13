#############
## imports ##
#############

import os
import argparse
from dataclasses import dataclass

#############
## classes ##
#############

@dataclass
class Hyperparams():
    # Class for storing the hyperparameter values
    total_epochs: int 
    test_after_n_epochs: int 
    batch_size: int 
    lr: float 
    z_dim: int
    adam_beta1: float
    adam_beta2: float

@dataclass
class DatasetParams():
    # Class for storing the dataset information
    dataset_path: str
    dataset_name: str
    img_size: int

class Params():
    def __init__(self):
        self.args = self.get_args()
        return

    def get_params(self):
        return self.get_hyperparams(), self.get_dataset_params()

    def get_hyperparams(self, verbose=True):
        hyperparams = Hyperparams(
            total_epochs=self.args.total_epochs,
            test_after_n_epochs=self.args.test_after_n_epochs,
            batch_size=self.args.batch_size,
            lr=self.args.learning_rate,
            adam_beta1=self.args.adam_beta1,
            adam_beta2=self.args.adam_beta2,
            z_dim=self.args.z_dim
        )
        if verbose:
            print('\tHyperparams values.')
            for field in hyperparams.__dataclass_fields__:
                value = getattr(hyperparams, field)
                print(f'\t\t{field} = {value}')
        return hyperparams

    def get_dataset_params(self, verbose=True):

        dataset_params = DatasetParams(
            dataset_path=self.args.dataset_path,
            dataset_name=self.args.dataset_name,
            img_size=self.args.img_size,
        )
        if verbose:
            print('\tDataset params')
            for field in dataset_params.__dataclass_fields__:
                value = getattr(dataset_params, field)
                print(f'\t\t{field} = {value}')
        return dataset_params


    def get_args(self):
        parser = argparse.ArgumentParser(description='Process some integers.')

        # Hyperparams
        parser.add_argument( '--total_epochs', type=int, default=200, help='Total epochs for the training.')
        parser.add_argument( '--batch_size', type=int, default=128, help='Batch size for the training phase.')
        parser.add_argument( '--learning_rate', type=float, default=2e-4, help='Learning rate value.')
        parser.add_argument( '--z_dim', type=int, default=100,help='Noise features.')
        parser.add_argument( '--adam_beta1', type=float, default=0.5,help='Adam beta 1.')
        parser.add_argument( '--adam_beta2', type=float, default=0.999,help='Adam beta 2.')
        parser.add_argument( '--test_after_n_epochs', type=int, default=10, help='Test the model after n epochs of training')
            
        # Dataset params
        parser.add_argument( '--dataset_path', type=str, default=os.getcwd()+'/datasets/', help='Path to the datasets folder.')
        parser.add_argument( '--dataset_name', type=str, default='mnist', help='Name of the dataset to be loaded for the training.')
        parser.add_argument( '--img_size', type=int, default=64, help='Size for squared images.')

        return parser.parse_args()

        return
