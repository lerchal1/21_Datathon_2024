from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import CIFAR10
import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from copy import copy


class BrandDataset(Dataset):
    def __init__(self, path, K=4, sequence_length=4):
        """Gets paths of training samples and outputs train/validation datasets with the same number of classes
            Args:
                path (str): Path to the csv file of the data
                K (int): Keep the last K elements for prediction
                sequence_length (int): length of the sequence o     

            Returns:
                tr, te: a tuple with the training paths in the first element and the validation paths in the second 
         """          
        # LoadCSV all at once
        self.path = pd.read_csv(path, sep=";").drop(["period_end_date,business_entity_doing_business_as_name"].values)
                
    def __getitem__(self, index):
        data = Image.open(os.path.join(self.path, self.data[index]))
        data = self.transform(data)
        label = torch.as_tensor(self.targets[index])
        return data, label
        
    def __len__(self):
        return len(self.data)

def get_balanced_sets(dataset, train_paths, test_size, age_ranges, age_from_path=lambda x: int(x.split('_')[0])):  
    """Gets paths of training samples and outputs train/validation datasets with the same number of classes

    Args:
        dataset (str): The name of the dataset
        train_paths (list): A list containing the paths to the training samples
        test_size (float): The size of the validation set   
        age_from_path (func): How to extract the age from the sample path         

    Returns:
        tr, te: a tuple with the training paths in the first element and the validation paths in the second 
    """                       
        
    tr, te = train_test_split(train_paths, test_size=test_size)
    while True:
        train_cl, test_cl = [], []
        for i in range(len(age_ranges) - 1):            
            from_ = age_ranges[i]
            to_ = age_ranges[i + 1]

            train_cl += [i for d in tr if from_ <= age_from_path(d) < to_]
            test_cl += [i for d in te if from_ <= age_from_path(d) < to_]
               
        if len(Counter(train_cl).keys()) == len(Counter(test_cl).keys()) == resolve_classes(dataset):
            break

        tr, te = train_test_split(train_paths, test_size=test_size)

    return tr, te


def get_datasets(name, path=None, test_size=0.0, augment=True, age_ranges=range(0, 101, 10)):
    """Produces the Dataset instances for training and validation from the name of the dataset

    Args:
        name (str): The name of the dataset
        path (list): The path of the dataset
        test_size (float): The size of the validation set
        augment (bool): Whether the training set needs augmentation or not            

    Returns:
        train, test: a tuple with the training paths in the first element and the validation paths in the second 
    """
    
    train_transform, test_transform = get_transforms(name)
    train, test = None, None

    if name == "cifar10":
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        
        train = CIFAR10(root= url, download=True, train=True, transform=train_transform) if augment else CIFAR10(root= url, download=True, train=True, transform=test_transform)
        test = CIFAR10(root= url, download=True, train=False, transform=test_transform)

    elif name == "utkface":
        assert path is not None, "expected to receive path to utkface images"
        
        train_paths = [p for p in os.listdir(path)]
        test_paths = []
        
        if test_size != 0.0: 
            train_paths, test_paths = get_balanced_sets(name, train_paths, test_size, age_ranges)
        
        train = FacesDataset(path, train_paths, train_transform) if augment else FacesDataset(path, train_paths, test_transform)
        test = FacesDataset(path, test_paths, test_transform)

    elif name == "fgnet":
        assert path is not None, "expected to receive path to fgnet images"
        
        train_paths = [p for p in os.listdir(path)]
        test_paths = []
        age = lambda x: int(x.replace('a', '').replace('b', '').split('.')[0][-2:])
        
        if test_size != 0.0: 
            train_paths, test_paths = get_balanced_sets(name, train_paths, test_size, age_from_path=age)
        
        train = FacesDataset(path, train_paths, train_transform, age_from_path=age) if augment else FacesDataset(path, train_paths, test_transform, age_from_path=age)
        test = FacesDataset(path, test_paths, test_transform, age_from_path=age)

    elif name == "morph":
        train_paths = [p for p in os.listdir(os.path.join(path, "Train"))]
        test_paths = [p for p in os.listdir(os.path.join(path, "Test"))]
        age = lambda x: int(x.split('.')[0][-2:])

        train = FacesDataset(os.path.join(path, "Train"), train_paths, train_transform, age_ranges, age_from_path=age) if augment else FacesDataset(path, train_paths, test_transform, age_ranges, age_from_path=age)
        test = FacesDataset(os.path.join(path, "Test"), test_paths, test_transform, age_ranges, age_from_path=age)

    elif name == "agedb":
        df = pd.read_csv(os.path.join(path, "AgeDB.csv"))        
        train_paths = [(p[5:], str(a).zfill(2) if a < 100 else 99) for a, p in df[df.folder != 0][["age", "img_path"]].values]
        test_paths = [(p[5:], str(a).zfill(2) if a < 100 else 99) for a, p in df[df.folder == 0][["age", "img_path"]].values]        
        
        train = AgeDB(path, train_paths, train_transform, age_ranges) if augment else AgeDB(path, train_paths, test_transform, age_ranges)
        test = AgeDB(path, test_paths, test_transform, age_ranges)


    else:
        raise Exception("Invalid dataset name")

    return train, test

def get_forget_retain_sets(dataset_name, dataset, forget_size=0.1, forget_indices=None, debug=False, augment_retain=False):
    """Gets a dataset and produces the forget and retain sets

    Args:
        dataset_name (str): The name of the dataset
        dataset (torch.utils.data.Dataset): The whole dataset
        forget_size (float): The size of the forget set
        forget_indices (list): If not None, contains a list with the indices of the forget set
        debug (bool): If true, print the forget set indices

    Returns:
        forget_set, retain_set: Subsets of the dataset that stand for the retain set and the forget set
    """

    if forget_indices is None:
        forget_len = int(forget_size * len(dataset))
        forget_indices = torch.randperm(len(dataset))[:forget_len]
    
    if debug:
        print(f"Forget indices: {forget_indices}")
    
    forget_set = torch.utils.data.Subset(copy(dataset), forget_indices) # Need to copy here because we are changing transform below
    forget_set.dataset.transform = eval(f"{dataset_name}_base_transform")

    retain_indices = [i for i in range(len(dataset)) if i not in forget_indices]
    retain_set = torch.utils.data.Subset(copy(dataset), retain_indices)
    if not augment_retain:
        retain_set.dataset.transform = eval(f"{dataset_name}_base_transform")


    return forget_set, retain_set