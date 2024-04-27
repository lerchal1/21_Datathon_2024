from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from copy import copy
from utils.metrics import *


def generate_data(df, window_length=10, K=4):
    nr_cols = len(df.columns) - 2

    inputs = torch.empty((0, window_length, nr_cols))
    labels = torch.empty((0, 1))
    
    # Iterate over col val
    for brand in df["business_entity_doing_business_as_name"].unique():
        df_filt = df[df["business_entity_doing_business_as_name"] == brand]
        # Drop useless columns
        values = df_filt.drop(["business_entity_doing_business_as_name", "period_end_date"], axis=1).values
        # For now cut-out first elements to have a perfect divisor
        values = values[(len(values) - K)%window_length:len(values) - K]
        for i in range(0, len(values) - window_length, window_length):
            input = torch.tensor(values[i:i + window_length]).unsqueeze(0)
            pred = growth_metric(torch.tensor(values[i+window_length:i+window_length+K])).unsqueeze(0).unsqueeze(0)
            inputs = torch.cat((inputs, input), 0)
            labels = torch.cat((labels, pred), 0)

    return inputs.to(torch.float32), labels.to(torch.float32)


class BrandDataset(Dataset):
    def __init__(self, path, window_length=10,  K=4):
        """Gets paths of training samples and outputs train/validation datasets with the same number of classes
            Args:
                path (str): Path to the csv file of the data
                K (int): Keep the last K elements for prediction
                window_length (int): length of the sequence o     

            Returns:
                tr, te: a tuple with the training paths in the first element and the validation paths in the second 
         """          
        # LoadCSV all at once
        print(path)
        self.df = pd.read_csv(path, sep=",")
        self.inputs, self.outputs = generate_data(self.df, window_length, K)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
        
    def __len__(self):
        return len(self.inputs)


def get_datasets(path=None, test_size=0.1):
    """Produces the Dataset instances for training and validation from the name of the dataset

    Args:
        name (str): The name of the dataset
        path (list): The path of the dataset
        test_size (float): The size of the validation set
        augment (bool): Whether the training set needs augmentation or not            

    Returns:
        train, test: a tuple with the training paths in the first element and the validation paths in the second 
    """
    
    dataset = BrandDataset(path)
    train, test = train_test_split(dataset, test_size=0.1)
    return train, test